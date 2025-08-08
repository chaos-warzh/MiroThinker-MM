# Copyright 2025 Miromind.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import argparse


def get_successful_log_paths(jsonl_file_path: str) -> list:
    """
    Extract log_file_path from records in JSONL file where final_judge_result is PASS_AT_K_SUCCESS

    Args:
        jsonl_file_path: Path to the JSONL file

    Returns:
        list: List of log_file_path
    """
    log_paths = []

    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if data.get("final_judge_result") == "PASS_AT_K_SUCCESS":
                        log_path = data.get("log_file_path")
                        if log_path:
                            log_paths.append(log_path)
                except json.JSONDecodeError:
                    continue

    return log_paths


# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract successful log paths from JSONL file"
    )
    parser.add_argument(
        "file_path", help="Path to the JSONL file containing benchmark results"
    )
    args = parser.parse_args()

    result = get_successful_log_paths(args.file_path)

    # Get the parent directory of args.file_path
    parent_dir = os.path.abspath(os.path.dirname(args.file_path))

    # Create successful logs directory
    success_log_dir = parent_dir + "/successful_logs"
    success_chatml_log_dir = parent_dir + "/successful_chatml_logs"
    os.makedirs(success_log_dir, exist_ok=True)
    print(f"Successful logs directory: {success_log_dir}")

    for i, path in enumerate(result, 1):
        basename = os.path.basename(path)
        print(f"Copying file: {path} to {success_log_dir}/{basename}")
        shutil.copy(path, f"{success_log_dir}/{basename}")

    os.system(
        f"uv run utils/converters/convert_to_chatml_auto_batch.py {success_log_dir}/*.json -o {success_chatml_log_dir}"
    )
    os.system(
        f"uv run utils/merge_chatml_msgs_to_one_json.py --input_dir {success_chatml_log_dir}"
    )
