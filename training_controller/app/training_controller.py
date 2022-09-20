# Standard Library
import asyncio
import json
import logging
import os
import time
from datetime import datetime

# Third Party
from elasticsearch import AsyncElasticsearch, exceptions
from elasticsearch.helpers import async_streaming_bulk
from fastapi import FastAPI
from opni_nats import NatsWrapper

# Opensearch config values
ES_ENDPOINT = os.environ["ES_ENDPOINT"]
ES_USERNAME = os.getenv("ES_USERNAME", "admin")
ES_PASSWORD = os.getenv("ES_PASSWORD", "admin")
DEFAULT_TRAINING_INTERVAL = 1800
GPU_TRAINING_RESET_TIME = 3600

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
nw = NatsWrapper()
es = AsyncElasticsearch(
    [ES_ENDPOINT],
    port=9200,
    http_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False,
    use_ssl=True,
)
gpu_training_request = 0
last_trainingjob_time = 0

es_instance = AsyncElasticsearch(
    [ES_ENDPOINT],
    port=9200,
    http_compress=True,
    http_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False,
    use_ssl=True,
)

app = FastAPI()


@app.get("/train_model")
async def train_model(workload_parameters: str):
    workload_parameters_dict = json.loads(workload_parameters)
    model_logs_query_body = {
        "query": {
            "bool": {
                "should": [],
            },
        },
    }
    for cluster_id in workload_parameters_dict:
        for namespace_name in workload_parameters_dict[cluster_id]:
            for pod_name in workload_parameters_dict[cluster_id][namespace_name]:
                bool_must_query = {
                    "bool": {
                        "must": [
                            {"match": {"cluster_id": cluster_id}},
                            {"match": {"kubernetes.namespace_name": namespace_name}},
                            {"match": {"kubernetes.pod_name": pod_name}},
                        ]
                    }
                }
                model_logs_query_body["query"]["bool"]["should"].append(bool_must_query)
    # This function handles get requests for fetching pod,namespace and workload breakdown insights.
    logging.info(f"Received request to train model.")
    try:
        await nw.publish("train", json.dumps(model_logs_query_body).encode())
        await nw.publish("workload_parameters", workload_parameters)
    except Exception as e:
        # Bad Request
        logging.error(e)


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
                "start_ts": (current_time - DEFAULT_TRAINING_INTERVAL) * (10**9),
                "end_ts": current_time * (10**9),
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
    """
    prepare the training data and launch nulog-train job
    """
    model_to_train = payload["model"]
    if model_to_train == "nulog-train":
        await nw.publish("gpu_trainingjob_status", b"JobStart")  # update gpu status
        await nw.publish(
            "gpu_service_training_internal", (json.dumps(payload)).encode()
        )  # schedule job


async def main():
    async def consume_nats_signal(msg):
        try:
            decoded_payload = json.loads(msg.data.decode())
            training_job_payload = {
                "source": "drain",
                "model": "nulog-train",
                "payload": decoded_payload,
            }
            await schedule_training_job(training_job_payload)
            logging.info("Just received signal to begin running the jobs")
            logging.info(decoded_payload)
        except Exception as e:
            logging.error(e)

    await nw.subscribe("train", subscribe_handler=consume_nats_signal)


async def consume_request():
    """
    consume incoming requests for inference on gpu-service.
    check the status of training jobs to prioritize them.
    inference requests get accepted only if there's no training jobs in queue.
    """

    async def gpu_available(msg):
        global gpu_training_request
        global last_trainingjob_time
        message = msg.data.decode()
        logging.info(f"message from training : {message}")
        if (
            message == "JobStart"
        ):  ## "JobStart" is published from frunction schedule_training_job()
            gpu_training_request += 1
            last_trainingjob_time = time.time()
        elif message == "JobEnd":  ## "JobEnd" is published from nulog-train contrainer
            gpu_training_request -= 1
        elif message == "JobReset":
            logging.info("Reset training job status.")
            gpu_training_request = 0

    async def receive_and_reply(msg):
        global last_trainingjob_time
        reply_subject = msg.reply
        if gpu_training_request > 0:
            if (
                last_trainingjob_time > 0
                and time.time() - last_trainingjob_time > GPU_TRAINING_RESET_TIME
            ):
                await nw.publish("gpu_trainingjob_status", b"JobReset")
                last_trainingjob_time = 0
            reply_message = b"NO"
        else:  ## gpu service available for inference
            await nw.publish("gpu_service_inference_internal", msg.data)
            reply_message = b"YES"
        logging.info(f"received inferencing request. response : {reply_message}")
        await nw.publish(reply_subject, reply_message)

    await nw.subscribe("gpu_trainingjob_status", subscribe_handler=gpu_available)
    await nw.subscribe("gpu_service_inference", subscribe_handler=receive_and_reply)


async def init_nats():
    logging.info("Attempting to connect to NATS")
    await nw.connect()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    task = loop.create_task(init_nats())
    loop.run_until_complete(task)
    consume_request_coroutine = consume_request()
    es_training_signal_coroutine = es_training_signal()

    main_coroutine = main()
    loop.run_until_complete(
        asyncio.gather(
            main_coroutine,
            consume_request_coroutine,
            es_training_signal_coroutine,
        )
    )
    try:
        loop.run_forever()
    finally:
        loop.close()
