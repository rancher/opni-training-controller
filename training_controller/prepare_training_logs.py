# Standard Library
import logging
import os
import shutil
import subprocess

# Third Party
import pandas as pd
from elasticsearch.helpers import scan

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
ES_ENDPOINT = os.environ["ES_ENDPOINT"]
ES_USERNAME = os.environ["ES_USERNAME"]
ES_PASSWORD = os.environ["ES_PASSWORD"]
FORMATTED_ES_ENDPOINT = (
    f"https://{ES_USERNAME}:{ES_PASSWORD}@" + ES_ENDPOINT.split("//")[-1]
)


class PrepareTrainingLogs:
    def __init__(self):
        self.WORKING_DIR = os.getenv("TRAINING_DATA_PATH", "/var/opni-data")
        self.TRAINING_DIR = os.path.join(self.WORKING_DIR, "windows")
        self.ES_DUMP_DIR = os.path.join(self.WORKING_DIR, "esdump_path")
        self.ES_DUMP_SAMPLE_LOGS_PATH = os.path.join(
            self.WORKING_DIR, "sample_logs.json"
        )

    def fetch_disk_size(self):
        # Fetch size of disk
        logging.info("Fetching size of the disk")
        total, used, free = shutil.disk_usage("/")
        logging.info("Disk Total: %d GiB" % (total // (2**30)))
        logging.info("Disk Used: %d GiB" % (used // (2**30)))
        logging.info("Disk Free: %d GiB" % (free // (2**30)))
        return free

    def run_esdump(self, query_commands):
        # Fetch the logs from each time interval within query_commands using Elasticdump.
        current_processes = set()
        max_processes = 2
        while len(query_commands) > 0:
            finished_processes = set()
            if len(current_processes) < max_processes:
                num_processes_to_run = min(
                    max_processes - len(current_processes), len(query_commands)
                )
                for i in range(num_processes_to_run):
                    current_query = query_commands.pop(0)
                    current_processes.add(
                        subprocess.Popen(
                            current_query, env={"NODE_TLS_REJECT_UNAUTHORIZED": "0"}
                        )
                    )
            for p in current_processes:
                if p.poll() is None:
                    p.wait()
                else:
                    finished_processes.add(p)
            current_processes -= finished_processes

    def retrieve_sample_logs_from_elasticsearch(self):
        # Get the last 10000 logs from Elasticsearch.

        logging.info("Retrieve sample logs from ES")
        es_dump_cmd = (
            'NODE_TLS_REJECT_UNAUTHORIZED=0 elasticdump --searchBody \'{"query": { "match_all": {} }, "_source": ["log", "time"], "sort": [{"time": {"order": "desc"}}]}\' --retryAttempts 10 --size=10000 --limit 10000 --input=%s/logs --output=%s --type=data'
            % (FORMATTED_ES_ENDPOINT, self.ES_DUMP_SAMPLE_LOGS_PATH)
        )
        subprocess.run(es_dump_cmd, shell=True)

        if os.path.exists(self.ES_DUMP_SAMPLE_LOGS_PATH):
            logging.info("Sampled downloaded successfully")
        else:
            logging.error("Sample failed to download")

    def calculate_training_logs_size(self, free):
        # Determine average size per log message
        sample_logs_bytes_size = os.path.getsize(self.ES_DUMP_SAMPLE_LOGS_PATH)
        num_lines = sum(1 for line in open(self.ES_DUMP_SAMPLE_LOGS_PATH))
        average_size_per_log_message = sample_logs_bytes_size / num_lines
        logging.info(
            f"average size per log message = {average_size_per_log_message} bytes"
        )
        os.remove(self.ES_DUMP_SAMPLE_LOGS_PATH)
        # Determine maximum number of logs to fetch for training
        num_logs_to_fetch = int((free * 0.8) / average_size_per_log_message)
        logging.info(f"Maximum number of log messages to fetch = {num_logs_to_fetch}")
        return num_logs_to_fetch

    def get_log_count(self, es_instance, timestamps_list, num_logs_to_fetch):
        # For each of the time intervals, fetch the number of logs present within that span.
        timestamps_esdump_num_logs_fetched = dict()
        total_number_of_logs = 0
        for timestamp_idx, timestamp_entry in enumerate(timestamps_list):
            start_ts, end_ts = timestamp_entry["start_ts"], timestamp_entry["end_ts"]
            query_body = {
                "query": {
                    "bool": {
                        "must": {"term": {"is_control_plane_log": "false"}},
                        "filter": [
                            {"range": {"timestamp": {"gte": start_ts, "lte": end_ts}}}
                        ],
                    }
                }
            }
            try:
                num_entries = es_instance.count(index="logs", body=query_body)["count"]
                timestamps_esdump_num_logs_fetched[timestamp_idx] = num_entries
                total_number_of_logs += num_entries
            except Exception as e:
                logging.error(e)
                continue
        total_number_of_logs_to_fetch = min(num_logs_to_fetch, total_number_of_logs)
        if total_number_of_logs > 0:
            for idx_key in timestamps_esdump_num_logs_fetched:
                timestamps_esdump_num_logs_fetched[idx_key] /= total_number_of_logs
                timestamps_esdump_num_logs_fetched[
                    idx_key
                ] *= total_number_of_logs_to_fetch
                timestamps_esdump_num_logs_fetched[idx_key] = int(
                    timestamps_esdump_num_logs_fetched[idx_key]
                )

        return timestamps_esdump_num_logs_fetched

    """
    def fetch_training_logs_from_elasticsearch(
        self, es_instance, num_logs_to_fetch, timestamps_list
    ):
        # This function will customize the Elasticdump query for each interval within timestamps_list and then fetch the logs by calling the run_esdump method.

        timestamps_esdump_num_logs_fetched = self.get_log_count(
            es_instance, timestamps_list, num_logs_to_fetch
        )
        # ESDump logs
        esdump_sample_command = [
            "/usr/local/bin/elasticdump",
            "--searchBody",
            '{{"query": {{"bool": {{"must": [{{"match":{{"drain_error_keyword":false}}}}, {{"term": {{"is_control_plane_log": false}}}},{{"range": {{"timestamp": {{"gte": {},"lt": {}}}}}}}]}}}} ,"_source": ["masked_log", "timestamp", "is_control_plane_log", "window_start_time_ns", "_id"], "sort": [{{"timestamp": {{"order": "desc"}}}}]}}',
            "--retryAttempts",
            "100",
            "--fileSize=50mb",
            "--size={}",
            "--limit",
            "10000",
            f"--input={FORMATTED_ES_ENDPOINT}/logs",
            "--output={}",
            "--type=data",
        ]
        query_queue = []
        for idx, entry in enumerate(timestamps_list):
            if timestamps_esdump_num_logs_fetched[idx] == 0:
                continue
            start_ts, end_ts, filename = (
                entry["start_ts"],
                entry["end_ts"],
                entry["filename"],
            )
            current_command = esdump_sample_command[:]
            current_command[2] = current_command[2].format(start_ts, end_ts)
            current_command[6] = current_command[6].format(
                timestamps_esdump_num_logs_fetched[idx]
            )
            current_command[10] = current_command[10].format(
                os.path.join(self.ES_DUMP_DIR, f"{filename}")
            )
            query_queue.append(current_command)
        # If at least one time interval within timestamps_list has a non zero amount of logs, call the es_dump command and return True
        if len(query_queue) > 0:
            self.run_esdump(query_queue)
            return True
        else:
            return False
    """

    def fetch_files_with_prefix(self, all_files, prefix):
        # Return the files within all_files which begin with prefix term.
        return [file for file in all_files if prefix in file]

    def delete_training_data_files(self, interval_json_files):
        # This function will remove all files listed in interval_json_files from self.TRAINING_DIR
        for interval_file in interval_json_files:
            os.remove(os.path.join(self.TRAINING_DIR, interval_file))

    def update_delete_interval_on_elasticsearch(
        self, es_instance, normal_interval, flag, updated_ts=None
    ):
        # Update or delete interval within opni-normal-intervals index depending on the specified flag, u for update or d for deletion.
        if flag == "u":
            try:
                es_instance.update(
                    index="opni-normal-intervals",
                    doc_type=normal_interval["_type"],
                    id=normal_interval["_id"],
                    body={"doc": {"start_ts": updated_ts}},
                )
                logging.info("Updating time interval within Elasticsearch.")
                return True
            except Exception as e:
                logging.error("Error updating document on opni-normal-intervals index.")
                return False
        elif flag == "d":
            # Delete the old time interval from Elasticsearch index opni-normal-intervals.
            try:
                es_instance.delete(
                    index="opni-normal-intervals",
                    doc_type=normal_interval["_type"],
                    id=normal_interval["_id"],
                )
                logging.info("Deleting old normal time interval from Elasticsearch")
                return True
            except Exception as e:
                logging.error(
                    "Error deleting document from opni-normal-intervals index."
                )
                return False
        else:
            logging.error("Invalid flag given!")
            return False

    def fetch_and_update_timestamps(self, es_instance):
        # This method will return a list of time intervals that have not already been fetched from Elasticsearch.

        # List that will store dictionaries with three keys: start_ts, end_ts and filename.
        timestamps_list = []
        try:
            # Obtain the oldest and newest logs from Elasticsearch
            oldest_log = es_instance.search(
                index="logs",
                body={
                    "aggs": {"min_ts": {"min": {"field": "timestamp"}}},
                    "_source": ["timestamp"],
                },
                size=1,
            )
            newest_log = es_instance.search(
                index="logs",
                body={
                    "aggs": {"max_ts": {"max": {"field": "timestamp"}}},
                    "_source": ["timestamp"],
                },
                size=1,
            )
        except Exception as e:
            logging.error(e)
            return timestamps_list
        # Retrieve the oldest and newest log timestamps.
        oldest_log_timestamp = int(oldest_log["aggregations"]["min_ts"]["value"])
        newest_log_timestamp = int(newest_log["aggregations"]["max_ts"]["value"])
        # Retrieve all the current normal training intervals from Elasticsearch.
        try:
            all_normal_intervals = scan(
                es_instance,
                index="opni-normal-intervals",
                query={"query": {"match_all": {}}},
            )
        except Exception as e:
            logging.error(
                "Error trying to retrieve all normal intervals from opni-normal-intervals index"
            )
            return timestamps_list
        # Retrieve all of the files currently stored in self.TRAINING_DIR
        all_training_files = os.listdir(self.TRAINING_DIR)
        for normal_interval in all_normal_intervals:
            start_ts, end_ts = (
                normal_interval["_source"]["start_ts"],
                normal_interval["_source"]["end_ts"],
            )
            full_file_prefix = f"{start_ts}_{end_ts}"

            # Fetch only the files within the all_training_files list which contain the full_file_prefix.
            interval_json_files = self.fetch_files_with_prefix(
                all_training_files, full_file_prefix
            )

            # If the end_ts is before the oldest_log_timestamp, then that interval should be removed from Elasticsearch as the data no longer exists.
            if end_ts < oldest_log_timestamp:
                # Delete interval within opni-normal-intervals index.
                is_success = self.update_delete_interval_on_elasticsearch(
                    es_instance, normal_interval, "d"
                )
                if is_success:
                    self.delete_training_data_files(interval_json_files)
                else:
                    continue
            # Address scenarios where start_ts comes before the oldest_log_timestamp
            elif start_ts < oldest_log_timestamp:
                """
                If end_ts is after the newest_log_timestamp, then set the start_ts to oldest_log_timestamp,
                keep the end_ts the same and set the filename to be named with the oldest_log_timestamp and
                newest_log_timestamp. Otherwise, set the filename to be named after the oldest_log_timestamp and end_ts.
                """
                if end_ts > newest_log_timestamp:
                    timestamps_list.append(
                        {
                            "start_ts": oldest_log_timestamp,
                            "end_ts": end_ts,
                            "filename": "{}_{}.json".format(
                                oldest_log_timestamp, newest_log_timestamp
                            ),
                        }
                    )
                else:
                    timestamps_list.append(
                        {
                            "start_ts": oldest_log_timestamp,
                            "end_ts": end_ts,
                            "filename": "{}_{}.json".format(
                                oldest_log_timestamp, end_ts
                            ),
                        }
                    )

                # Update start_ts value to oldest_log_timestamp.
                is_success = self.update_delete_interval_on_elasticsearch(
                    es_instance, normal_interval, "u", oldest_log_timestamp
                )
                if is_success:
                    logging.info(
                        "Removing old JSON training files where the time interval within opni-normal-intervals was updated."
                    )
                    self.delete_training_data_files(interval_json_files)
                else:
                    continue

            # Address scenario where start_ts is on or after oldest_log_timestamp and end_ts is after newest_log_timestamp.
            elif end_ts > newest_log_timestamp:
                # Set the filename to be named with the start_ts amd newest_log_timestamp
                timestamps_list.append(
                    {
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "filename": f"{start_ts}_{newest_log_timestamp}.json",
                    }
                )

            # Address scenario where start_ts is on or after oldest_log_timestamp and end_ts is before or on newest_log_timestamp.
            else:
                # If there already exist files with the filename prefix, do not fetch that data again.
                if len(interval_json_files) == 0:
                    # Fetch the files within self.TRAINING_DIR that contain just the start_ts in its name
                    start_ts_interval_json_files = self.fetch_files_with_prefix(
                        all_training_files, str(start_ts)
                    )
                    logging.info(
                        "Removing old JSON training files with the same starting timestamp but updated ending timestamp."
                    )
                    self.delete_training_data_files(start_ts_interval_json_files)
                    timestamps_list.append(
                        {
                            "start_ts": start_ts,
                            "end_ts": end_ts,
                            "filename": f"{start_ts}_{end_ts}.json",
                        }
                    )

        return timestamps_list

    def normalize_json_data(self):
        # For every json file obtained through Elasticdump, normalize the _source field and dump that result into the self.TRAINING_DIR directory.
        for es_split_json_file in os.listdir(self.ES_DUMP_DIR):
            if not ".json" in es_split_json_file:
                continue
            json_file_to_process = os.path.join(self.ES_DUMP_DIR, es_split_json_file)
            df = pd.read_json(json_file_to_process, lines=True)
            df = pd.json_normalize(df["_source"])
            df[
                [
                    "timestamp",
                    "window_start_time_ns",
                    "masked_log",
                    "is_control_plane_log",
                ]
            ].to_json(
                os.path.join(
                    self.TRAINING_DIR,
                    "{}.json.gz".format(es_split_json_file.split(".json")[0]),
                ),
                orient="records",
                lines=True,
                compression="gzip",
            )
            # delete ESDumped file
            os.remove(json_file_to_process)
        # Delete the ES_DUMP_DIR as well.
        shutil.rmtree(self.ES_DUMP_DIR)

    """
    def run(self):
        if not os.path.exists(self.ES_DUMP_DIR):
            os.makedirs(self.ES_DUMP_DIR)

        if not os.path.exists(self.TRAINING_DIR):
            os.makedirs(self.TRAINING_DIR)
        es_instance = Elasticsearch(
            [ES_ENDPOINT],
            port=9200,
            http_auth=(ES_USERNAME, ES_PASSWORD),
            verify_certs=False,
            use_ssl=True,
        )
        free = self.fetch_disk_size()
        self.retrieve_sample_logs_from_elasticsearch()
        num_logs_to_fetch = self.calculate_training_logs_size(free)
        timestamps_list = self.fetch_and_update_timestamps(es_instance)
        data_exists = self.fetch_training_logs_from_elasticsearch(
            es_instance, num_logs_to_fetch, timestamps_list
        )
        if data_exists:
            self.normalize_json_data()
        return data_exists
    """

    def get_num_logs_for_training(self):
        if not os.path.exists(self.ES_DUMP_DIR):
            os.makedirs(self.ES_DUMP_DIR)

        if not os.path.exists(self.TRAINING_DIR):
            os.makedirs(self.TRAINING_DIR)
        free = self.fetch_disk_size()
        self.retrieve_sample_logs_from_elasticsearch()
        num_logs_to_fetch = self.calculate_training_logs_size(free)
        return num_logs_to_fetch
