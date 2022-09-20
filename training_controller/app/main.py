# Standard Library
import asyncio
import json
import logging
import os
import time

# Third Party
import boto3
from botocore.client import Config
from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI
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

app = FastAPI()
nw = NatsWrapper()
GPU_TRAINING_RESET_TIME = 3600
gpu_training_request = 0
last_trainingjob_time = 0
workload_parameters_dict = dict()


def get_namespace_deployment_breakdown(pod_aggregation_data):
    # Get the breakdown of log messages by namespace and pod name.
    ns_pod_aggregation_dict = dict()
    try:
        for batch in pod_aggregation_data:
            for each_result in batch:
                namespace_name, pod_name, doc_count = (
                    each_result["key"]["namespace_name"],
                    each_result["key"]["deployment_name"],
                    each_result["doc_count"],
                )
                if not namespace_name in ns_pod_aggregation_dict:
                    ns_pod_aggregation_dict[namespace_name] = dict()
                if not pod_name in ns_pod_aggregation_dict[namespace_name]:
                    ns_pod_aggregation_dict[namespace_name][pod_name] = doc_count
        return ns_pod_aggregation_dict
    except Exception as e:
        logging.error(f"Unable to aggregate pod data. {e}")
        return ns_pod_aggregation_dict


async def get_aggregated_data(query_body):
    # Given an Elasticsearch query, fetch all results using composite aggregation.
    all_aggregation_results = []
    while True:
        try:
            aggregation_results = await es_instance.search(
                index="logs", body=query_body
            )
            # If an after_key is present, use it to fetch the next batch of aggregations from Elasticsearch. Otherwise, return all aggregations.
            if "after_key" in aggregation_results["aggregations"]["bucket"]:
                after_key = aggregation_results["aggregations"]["bucket"]["after_key"]
                query_body["aggs"]["bucket"]["composite"]["after"] = after_key
                all_aggregation_results.append(
                    aggregation_results["aggregations"]["bucket"]["buckets"]
                )
            else:
                return all_aggregation_results
        except Exception as e:
            logging.error(e)
            return all_aggregation_results


@app.get("/storage_space")
async def get_storage_space():
    return PrepareTrainingLogs().get_num_logs_for_training()


@app.get("/train_model")
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
        logging.info(max_logs_for_training)
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
            return "submitted request to train model"
        except Exception as e:
            # Bad Request
            logging.error(e)
    else:
        return "currently unable to train model currently. please try again later."


@app.get("/get_workloads")
async def get_workloads(cluster_id: str):
    other_query_body = {
        "size": 0,
        "query": {
            "bool": {
                "must": [
                    {"match": {"cluster_id": cluster_id}},
                    {"match": {"log_type": "workload"}},
                    {"regexp": {"kubernetes.namespace_name.keyword": ".+"}},
                    {"regexp": {"deployment.keyword": ".+"}},
                ],
            }
        },
        "aggs": {
            "bucket": {
                "composite": {
                    "size": 1000,
                    "sources": [
                        {
                            "namespace_name": {
                                "terms": {"field": "kubernetes.namespace_name.keyword"}
                            }
                        },
                        {"deployment_name": {"terms": {"field": "deployment.keyword"}}},
                    ],
                }
            }
        },
    }
    try:
        pod_aggregation_data = await get_aggregated_data(other_query_body)
        logging.info(pod_aggregation_data)
        namespace_pod_breakdown_dict = get_namespace_deployment_breakdown(
            pod_aggregation_data
        )
        return namespace_pod_breakdown_dict
    except Exception as e:
        logging.error(f"Unable to breakdown pod insights. {e}")
        return {}


@app.get("/get_model_parameters")
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


@app.get("/get_model_status")
async def get_model_status():
    if gpu_training_request > 0:
        return "training"
    else:
        if verify_model_saved():
            return "completed"
        else:
            return "not started"


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


@app.on_event("startup")
async def startup_event():
    logging.info("Here we are on startup!!!!")
    init_nats_task = asyncio.create_task(init_nats())
    await init_nats_task
    workload_task = asyncio.create_task(get_latest_workload())
    await workload_task
    await asyncio.gather(main(), consume_request())
