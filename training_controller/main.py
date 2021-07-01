# Standard Library
import asyncio
import logging
import os
from datetime import datetime

# Third Party
from elasticsearch import AsyncElasticsearch, exceptions
from elasticsearch.helpers import async_streaming_bulk
from opni_nats import NatsWrapper
from prepare_training_logs import PrepareTrainingLogs

ES_ENDPOINT = os.environ["ES_ENDPOINT"]
ES_USERNAME = os.getenv("ES_USERNAME", "admin")
ES_PASSWORD = os.getenv("ES_PASSWORD", "admin")
MINIO_SERVER_URL = os.environ["MINIO_SERVER_URL"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
NATS_SERVER_URL = os.environ["NATS_SERVER_URL"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

DEFAULT_TRAINING_INTERVAL = 1800

nw = NatsWrapper()

es = AsyncElasticsearch(
    [ES_ENDPOINT],
    port=9200,
    http_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False,
    use_ssl=True,
)


async def update_es_job_status(
    request_id: str,
    job_status: str,
    op_type: str = "update",
    index: str = "training_signal",
):
    """
    this method updates the status of jobs in elasticsearch.
    """
    script = f"ctx._source.status = '{job_status}';"
    docs_to_update = [
        {
            "_id": request_id,
            "_op_type": op_type,
            "_index": index,
            "script": script,
        }
    ]
    logging.info(f"ES job {request_id} status update : {job_status}")
    try:
        async for ok, result in async_streaming_bulk(es, docs_to_update):
            action, result = result.popitem()
            if not ok:
                logging.error(f"failed to index documents {docs_to_update}")
    except Exception as e:
        logging.error(e)


async def es_training_signal():
    """
    collect job training signal from elasticsearch, and add to job queue.
    """
    query_body = {"query": {"bool": {"must": {"match": {"status": "submitted"}}}}}
    index = "training_signal"
    current_time = int(datetime.timestamp(datetime.now()))
    job_payload = {
        "model_to_train": "nulog",
        "time_intervals": [
            {
                "start_ts": (current_time - DEFAULT_TRAINING_INTERVAL) * (10 ** 9),
                "end_ts": current_time * (10 ** 9),
            }
        ],
    }
    signal_index_exists = False
    try:
        signal_index_exists = await es.indices.exists(index)
        if not signal_index_exists:
            signal_created = await es.indices.create(index=index)
    except exceptions.TransportError as e:
        logging.error(e)
    while True:
        try:
            user_signals_response = await es.search(
                index=index, body=query_body, size=100
            )
            user_signal_hits = user_signals_response["hits"]["hits"]
            if len(user_signal_hits) > 0:
                for hit in user_signal_hits:
                    training_job_payload = {
                        "source": "elasticsearch",
                        "_id": hit["_id"],
                        "model": "nulog-train",
                        "payload": job_payload,
                    }
                    await update_es_job_status(
                        request_id=hit["_id"], job_status="training scheduled"
                    )
                    await schedule_training_job(training_job_payload)
        except (exceptions.NotFoundError, exceptions.TransportError) as e:
            logging.error(e)

        await asyncio.sleep(60)


async def schedule_training_job(payload):
    model_to_train = payload["model"]
    model_payload = payload["payload"]
    PrepareTrainingLogs("/tmp").run(model_payload["time_intervals"])
    if model_to_train == "nulog-train":
        # if "source" in payload and payload["source"] == "elasticsearch":
        #     await update_es_job_status(
        #         request_id=payload["_id"], job_status="trainingStarted"
        #     )
        await nw.publish(
            "gpu_available", b"JobAdded"
        )  # controller will start to decline inference requests
        await nw.publish(
            "gpu_service_training_internal", pd.to_json(payload)
        )  # send jobs to nulog-train container


async def main(request_queue):
    async def consume_nats_signal(msg):
        try:
            decoded_payload = pd.read_json(msg.data().encode())  ## to fix
            # process the payload
            if decoded_payload["model_to_train"] == "nulog":
                training_job_payload = {
                    "source": "drain",
                    "model": "nulog-train",
                    "payload": decoded_payload,
                }
                await schedule_training_job(training_job_payload)
            logging.info("Just received signal to begin running the jobs")

        except Exception as e:
            logging.error(e)

    await nw.nc.subscribe(nats_subject="train", cb=consume_nats_signal)

    #     ##TODO: move to nulog-train
    #     await minio_setup_and_download_data(minio_client) # download from minio
    #     await model_trained = train_nulog_model(minio_client, "windows/") # training job
    #     if model_trained: ## assume the model training always works
    #         gpu_training_request -= 1
    #         # notify inference services.


async def consume_request():
    gpu_training_request = 0

    async def gpu_available(msg):
        message = msg.data().decode()
        if message == "JobAdded":
            gpu_training_request += 1
        elif message == "JobComplete":
            gpu_training_request -= 1

    async def receive_and_reply(msg):
        reply_subject = msg.reply

        if gpu_training_request == 0:
            nw.publish(
                nats_subject="gpu_service_inference_internal", payload=msg
            )  ## to fix
            reply_message = b"request accepted."
        else:
            reply_message = b"request declined."
        await nw.nc.publish(reply_subject, reply_message)

    await nw.nc.subscribe(nats_subject="gpu_available", cb=gpu_available)
    await nw.nc.subscribe(nats_subject="gpu_service_inference", cb=receive_and_reply)


async def init_nats():
    logging.info("Attempting to connect to NATS")
    await nw.connect()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    task = loop.create_task(init_nats())
    loop.run_until_complete(task)
    consume_request_coroutine = consume_request()
    main_coroutine = main()

    loop.run_until_complete(asyncio.gather(main_coroutine, consume_request_coroutine))
    try:
        loop.run_forever()
    finally:
        loop.close()
