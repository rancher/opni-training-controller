# Standard Library
import asyncio
import json
import logging
import os
import time

# Third Party
import boto3
import requests
from botocore.client import Config
from elasticsearch import AsyncElasticsearch
from nats.aio.errors import ErrTimeout
from opni_nats import NatsWrapper
from prepare_training_logs import PrepareTrainingLogs

ES_ENDPOINT = os.environ["ES_ENDPOINT"]
ES_USERNAME = os.getenv("ES_USERNAME", "admin")
ES_PASSWORD = os.getenv("ES_PASSWORD", "admin")
# s3 config values
S3_ACCESS_KEY = os.environ["S3_ACCESS_KEY"]
S3_SECRET_KEY = os.environ["S3_SECRET_KEY"]
S3_ENDPOINT = os.environ["S3_ENDPOINT"]
S3_BUCKET = os.getenv("S3_BUCKET", "opni-nulog-models")

es_instance = AsyncElasticsearch(
    [ES_ENDPOINT],
    port=9200,
    http_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False,
    use_ssl=True,
)

s3_client = boto3.resource(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version="s3v4"),
)

nw = NatsWrapper()
GPU_TRAINING_RESET_TIME = 3600
gpu_training_request = 0
last_trainingjob_time = 0
workload_parameters_dict = dict()


def get_gpu_status():
    try:
        results = requests.get("http://opni-internal:11080/ModelTraining/gpu_info")
        decoded_result = json.loads(results.content.decode())
        if "list" in decoded_result:
            gpu_resources_list = decoded_result["list"]
            for gpu_info in gpu_resources_list:
                if int(gpu_info["allocatable"]) > 0:
                    return "available"
        return "unavailable"

    except Exception as e:
        logging.error(e)
        return "unavailable"


async def get_gpu_service_status():
    try:
        response = await nw.request("gpu_service_running", b"check-up", timeout=1)
        return response.data.decode()
    except ErrTimeout:
        return "unavailable"


async def train_model(workload_parameters: str):
    if gpu_training_request == 0:
        global workload_parameters_dict
        workload_parameters_dict = json.loads(workload_parameters)
        current_ts = int(time.time() * 1000)
        await es_instance.index(
            index="model-training-parameters",
            body={"time": current_ts, "parameters": workload_parameters},
        )
        max_logs_for_training = PrepareTrainingLogs().get_num_logs_for_training()
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - 3600000
        model_logs_query_body = {
            "query": {
                "bool": {
                    "filter": [{"range": {"time": {"gte": start_ts, "lte": end_ts}}}],
                    "minimum_should_match": 1,
                    "should": [],
                }
            }
        }
        for cluster_id in workload_parameters_dict:
            for namespace_name in workload_parameters_dict[cluster_id]:
                for deployment_name in workload_parameters_dict[cluster_id][
                    namespace_name
                ]:
                    should_query_string = {
                        "query_string": {
                            "fields": [
                                "cluster_id",
                                "kubernetes.namespace_name.keyword",
                                "deployment.keyword",
                            ],
                            "query": f"{cluster_id} AND {namespace_name} AND {deployment_name}",
                        }
                    }
                    model_logs_query_body["query"]["bool"]["should"].append(
                        should_query_string
                    )
        # This function handles get requests for fetching pod,namespace and workload breakdown insights.
        logging.info(f"Received request to train model.")
        payload_query = {
            "max_size": max_logs_for_training,
            "query": model_logs_query_body,
        }
        try:
            await nw.publish("workload_parameters", workload_parameters.encode())
            await nw.publish("train", json.dumps(payload_query).encode())
            return b"submitted request to train model"
        except Exception as e:
            # Bad Request
            logging.error(e)
            return b"currently unable to train model. please try again later"
    else:
        return b"currently unable to train model. please try again later."


async def get_model_parameters():
    return workload_parameters_dict


def verify_model_saved():
    bucket = s3_client.Bucket("opni-nulog-models")
    model_file = "nulog_model_latest.pt"
    try:
        objs = list(bucket.objects.filter(Prefix=model_file))
        return len(objs) > 0
    except Exception as e:
        logging.error(e)
        return False


async def get_model_status():
    if gpu_training_request > 0:
        return b"training"
    else:
        if verify_model_saved():
            return b"completed"
        else:
            return b"not started"


async def get_latest_workload():
    global workload_parameters_dict
    query_body = {"sort": [{"time": {"order": "desc"}}], "query": {"match_all": {}}}
    try:
        latest_workload = await es_instance.search(
            index="model-training-parameters", body=query_body, size=1
        )
        workload_parameters_dict = json.loads(
            latest_workload["hits"]["hits"][0]["_source"]["parameters"]
        )
    except Exception as e:
        logging.error(e)


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
        ):  ## "JobStart" is published from function schedule_training_job()
            gpu_training_request += 1
            last_trainingjob_time = time.time()
        elif message == "JobEnd":
            gpu_training_request -= 1

    async def receive_and_reply(msg):
        global last_trainingjob_time
        reply_subject = msg.reply
        gpu_status = get_gpu_status()
        gpu_service_status = await get_gpu_service_status()
        if (
            gpu_training_request > 0
            or gpu_status == "unavailable"
            or gpu_service_status == "unavailable"
        ):
            reply_message = b"NO"
        else:  ## gpu service available for inference
            await nw.publish("gpu_service_inference_internal", msg.data)
            reply_message = b"YES"
        logging.info(f"received inferencing request. response : {reply_message}")
        await nw.publish(reply_subject, reply_message)

    await nw.subscribe("gpu_trainingjob_status", subscribe_handler=gpu_available)
    await nw.subscribe("gpu_service_inference", subscribe_handler=receive_and_reply)


async def endpoint_backends():
    async def model_status_sub_handler(msg):
        reply_subject = msg.reply
        reply_message = await get_model_status()
        await nw.publish(reply_subject, reply_message)

    async def workload_parameters_sub_handler(msg):
        reply_subject = msg.reply
        reply_message = json.dumps(workload_parameters_dict).encode()
        await nw.publish(reply_subject, reply_message)

    async def train_model_sub_handler(msg):
        reply_subject = msg.reply
        training_payload = msg.data.decode()
        await nw.publish(reply_subject, b"training job submitted")
        reply_message = await train_model(training_payload)

    await nw.subscribe("model_status", subscribe_handler=model_status_sub_handler)
    await nw.subscribe(
        "workload_parameters", subscribe_handler=workload_parameters_sub_handler
    )
    await nw.subscribe("train_model", subscribe_handler=train_model_sub_handler)


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


async def init_nats():
    logging.info("Attempting to connect to NATS")
    await nw.connect()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    task = loop.create_task(init_nats())
    loop.run_until_complete(task)
    consume_request_coroutine = consume_request()
    plugin_backends_coroutine = endpoint_backends()
    latest_workload_coroutine = get_latest_workload()
    main_coroutine = main()
    loop.run_until_complete(
        asyncio.gather(
            main_coroutine,
            consume_request_coroutine,
            plugin_backends_coroutine,
            latest_workload_coroutine,
        )
    )
    try:
        loop.run_forever()
    finally:
        loop.close()
