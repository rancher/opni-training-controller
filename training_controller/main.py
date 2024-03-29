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
from opni_nats import NatsWrapper
from prepare_training_logs import PrepareTrainingLogs

ES_ENDPOINT = os.environ["ES_ENDPOINT"]
ES_USERNAME = os.environ["ES_USERNAME"]
ES_PASSWORD = os.environ["ES_PASSWORD"]
# s3 config values
S3_ACCESS_KEY = os.environ["S3_ACCESS_KEY"]
S3_SECRET_KEY = os.environ["S3_SECRET_KEY"]
S3_ENDPOINT = os.environ["S3_ENDPOINT"]
S3_BUCKET = os.getenv("S3_BUCKET", "opni-nulog-models")
MODEL_STATS_ENDPOINT = "http://opni-internal:11080/ModelTraining/model/current_status"
RETRY_LIMIT = 15

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
GPU_GATEWAY_ENDPOINT = "http://opni-internal:11080/ModelTraining/gpu_info"
# unit: ms. With introducing streaming data loader, it's possible to download much more training data.
TRAINING_DATA_INTERVAL = 3600 * 1000 * 1
ANOMALY_KEYWORDS = [
    "error",
    "fail",
    "fatal",
    "exception",
    "timeout",
    "unavailable",
    "crash",
    "connection refused",
    "network error",
    "deadlock",
    "out of disk",
    "high load",
]


def post_model_status(status):
    model_training_status = {"status": status}
    try:
        result = requests.put(
            MODEL_STATS_ENDPOINT, data=json.dumps(model_training_status).encode(),
            headers={"Content-Type": "application/json", "Accept" : "application/json"},
        )
        logging.info(f"Posted training status, result: {result}")
    except Exception as e:
        logging.warning(f"Failed to post training status, error: {e}")


def get_gpu_status():
    try:
        results = requests.get(
            GPU_GATEWAY_ENDPOINT,
            headers={"Content-Type": "application/json", "Accept" : "application/json"},
        )
        decoded_result = json.loads(results.content.decode())
        if "items" in decoded_result:
            gpu_resources_list = decoded_result["items"]
            for gpu_info in gpu_resources_list:
                if int(gpu_info["allocatable"]) > 0:
                    return "available"
        return "unavailable"

    except Exception as e:
        logging.error(e)
        return "unavailable"


async def get_gpu_service_status():
    """
    get_gpu_service_status will either return running or unavailable.
    It will return running if the GPU service pod is up and running and sends back a response.
    Otherwise, if there is a timeout, then it will return unavailable.
    """
    try:
        response = await nw.request("gpu_service_running", b"check-up", timeout=5)
        return response.data.decode()
    except Exception as e:
        logging.error(e)
        return "unavailable"


async def get_nats_bucket_kv():
    result_dict = dict()
    model_training_bucket = await nw.get_bucket("model-training-parameters")
    current_bucket_payload = await model_training_bucket.get("modelTrainingParameters")
    last_trained_bucket_payload = await model_training_bucket.get("lastModelTrained")
    result_dict["current_workload_parameters"] = json.loads(
        current_bucket_payload.decode()
    )
    if last_trained_bucket_payload:
        last_trained_payload_dict = json.loads(last_trained_bucket_payload.decode())
        if "payload" in last_trained_payload_dict:
            result_dict["last_trained_workload_parameters"] = last_trained_payload_dict[
                "payload"
            ]["parameters"]
    return result_dict


def check_training_necessary(
    previous_workload_parameters_dict: dict, current_workload_parameters_dict: dict
):
    for cluster_id in current_workload_parameters_dict["workloads"]:
        if not cluster_id in previous_workload_parameters_dict["workloads"]:
            return True
        for namespace_name in current_workload_parameters_dict["workloads"][cluster_id]:
            if (
                not namespace_name
                in previous_workload_parameters_dict["workloads"][cluster_id]
            ):
                return True
            for deployment_name in current_workload_parameters_dict["workloads"][
                cluster_id
            ][namespace_name]:
                if (
                    not deployment_name
                    in previous_workload_parameters_dict["workloads"][cluster_id][
                        namespace_name
                    ]
                ):
                    return True
    return False


async def schedule_model_training():
    model_training_bucket_dict = await get_nats_bucket_kv()
    current_workload_parameters_dict = model_training_bucket_dict[
        "current_workload_parameters"
    ]
    model_training_necessary = True
    if "last_trained_workload_parameters" in model_training_bucket_dict:
        last_trained_payload_dict = model_training_bucket_dict[
            "last_trained_workload_parameters"
        ]
        if "workloads" in last_trained_payload_dict:
            model_training_necessary = check_training_necessary(
                last_trained_payload_dict, current_workload_parameters_dict
            )
    workload_parameter_payload = {
        "workloads": current_workload_parameters_dict["workloads"]
    }

    if model_training_necessary:
        gpu_service_status = await get_gpu_service_status()
        num_retries = 0
        while gpu_service_status != "running" and num_retries < RETRY_LIMIT:
            await asyncio.sleep(30)
            gpu_service_status = await get_gpu_service_status()
            num_retries += 1
        if gpu_service_status == "running":
            await train_model(current_workload_parameters_dict)
            workload_parameter_payload["status_type"] = "train"
            await nw.publish(
                "model_workload_parameters",
                json.dumps(workload_parameter_payload).encode(),
            )
        else:
            logging.info(
                "GPU service is currently unavailable so model cannot be trained."
            )
            post_model_status(status="training failed")

    else:
        workload_parameter_payload["status_type"] = "update"
        await nw.publish(
            "model_workload_parameters", json.dumps(workload_parameter_payload).encode()
        )
        logging.info("Workload parameters have been updated for inferencing.")
        post_model_status(status="completed")


async def train_model(workload_parameters_dict):
    if gpu_training_request == 0:
        max_logs_for_training = PrepareTrainingLogs().get_num_logs_for_training()
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - TRAINING_DATA_INTERVAL
        parentheses_keywords = [f"({x})" for x in ANOMALY_KEYWORDS]
        workload_parameters = workload_parameters_dict["workloads"]
        model_logs_query_body = {
            "query": {
                "bool": {
                    "filter": [{"range": {"time": {"gte": start_ts, "lte": end_ts}}}],
                    "minimum_should_match": 1,
                    "should": [],
                    "must_not": [
                        {"match": {"anomaly_level.keyword": "Anomaly"}},
                        {
                            "query_string": {
                                "query": " or ".join(parentheses_keywords),
                                "default_field": "log",
                            }
                        },
                    ],
                },
            }
        }
        for cluster_id in workload_parameters:
            for namespace_name in workload_parameters[cluster_id]:
                for deployment_name in workload_parameters[cluster_id][namespace_name]:
                    should_query_string = {
                        "query_string": {
                            "fields": [
                                "cluster_id",
                                "namespace_name.keyword",
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
        training_data_count = (
            await es_instance.count(index="logs", body=model_logs_query_body)
        )["count"]
        payload_query = {
            "max_size": max_logs_for_training,
            "query": model_logs_query_body,
            "count": training_data_count,
            "parameters": workload_parameters_dict,
        }
        try:
            await nw.publish("train", json.dumps(payload_query).encode())
            logging.info(f"payload : {payload_query}")
            return b"submitted request to train model"
        except Exception as e:
            # Bad Request
            logging.error(e)
            return b"currently unable to train model. please try again later"
    else:
        return b"currently unable to train model. please try again later."


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


async def get_workloads_from_nats():
    """
    This function will fetch the current model training parameters as well as the last model training parameters from
    Nats kv. It will then check to see if the UUIDs are different between the current parameters and the parameters from
     the last model trained. If they are not the same, it will send it over to the training function.
    """
    try:
        model_training_bucket_dict = await get_nats_bucket_kv()
        workload_parameters_dict = model_training_bucket_dict[
            "current_workload_parameters"
        ]
        if "last_trained_workload_parameters" in model_training_bucket_dict:
            last_trained_payload = model_training_bucket_dict[
                "last_trained_workload_parameters"
            ]
            if workload_parameters_dict["uuid"] != last_trained_payload["uuid"] and (
                "workloads" in workload_parameters_dict
            ):
                await schedule_model_training()
        else:
            await schedule_model_training()
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

    async def train_reset_model_sub_handler(msg):
        reply_subject = msg.reply
        training_payload = json.loads(msg.data.decode())
        if "workloads" in training_payload:
            await nw.publish(reply_subject, b"training job submitted")
            await schedule_model_training()
        else:
            await nw.publish(reply_subject, b"model reset")
            model_reset_payload = {"status": "reset"}
            await nw.publish("model_update", json.dumps(model_reset_payload).encode())
            reset_payload = {"workloads": {}, "status_type": "reset"}
            await nw.publish(
                "model_workload_parameters", json.dumps(reset_payload).encode()
            )
            model_training_parameters_bucket = await nw.get_bucket(
                "model-training-parameters"
            )
            operation = await model_training_parameters_bucket.put(
                "lastModelTrained", json.dumps({}).encode()
            )

    await nw.subscribe("model_status", subscribe_handler=model_status_sub_handler)
    await nw.subscribe("train_model", subscribe_handler=train_reset_model_sub_handler)


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
    latest_workload_coroutine = get_workloads_from_nats()
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
