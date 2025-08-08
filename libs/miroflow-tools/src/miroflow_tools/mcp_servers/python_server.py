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

import os

from e2b_code_interpreter import Sandbox
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("e2b-python-interpreter")

# API keys
E2B_API_KEY = os.environ.get("E2B_API_KEY")

# DEFAULT CONFS
DEFAULT_TIMEOUT = 300  # seconds


@mcp.tool()
async def create_sandbox(timeout: int = DEFAULT_TIMEOUT) -> str:
    """Create a linux sandbox.

    Args:
        timeout: Time in seconds before the sandbox is automatically shutdown. The default is 300 seconds.

    Returns:
        The id of the newly created sandbox. You should use this sandbox_id to run other tools in the sandbox.
    """
    sandbox = Sandbox(timeout=timeout, api_key=E2B_API_KEY)
    info = sandbox.get_info()
    return f"sandbox_id: {info.sandbox_id}"


@mcp.tool()
async def run_command(command: str, sandbox_id: str) -> str:
    """Execute a command in the linux sandbox.

    Args:
        command: The command to execute
        sandbox_id: The id of the sandbox to execute the command in. To create a new sandbox, use tool `create_sandbox`.

    Returns:
        A CommandResult object containing the result of the command execution, format like CommandResult(stderr=..., stdout=..., exit_code=..., error=...)
    """
    sandbox = Sandbox.connect(sandbox_id, api_key=E2B_API_KEY)
    sandbox.set_timeout(
        DEFAULT_TIMEOUT
    )  # refresh the timeout for each command execution
    try:
        result = sandbox.commands.run(command)
        return str(result)
    except Exception as e:
        return f"Failed to run command: {e}"


@mcp.tool()
async def run_python_code(code_block, sandbox_id: str) -> str:
    """Run python code in an interpreter and return the execution result.

    Args:
        code_block: The python code to run.
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.

    Returns:
        A CommandResult object containing the result of the command execution, format like CommandResult(stderr=..., stdout=..., exit_code=..., error=...)
    """

    template = None
    if template:
        try:
            sandbox = Sandbox.connect(sandbox_id=sandbox_id, api_key=E2B_API_KEY)
        except Exception:
            return f"Failed to connect to sandbox {sandbox_id}"
    else:
        try:
            sandbox = Sandbox.connect(sandbox_id=sandbox_id, api_key=E2B_API_KEY)
        except Exception:
            return f"Failed to connect to sandbox {sandbox_id}"

    sandbox.set_timeout(
        DEFAULT_TIMEOUT
    )  # refresh the timeout for each command execution

    try:
        execution = sandbox.run_code(code_block)
        return str(execution)
    except Exception as e:
        return f"Failed to run code: {e}"


@mcp.tool()
async def upload_local_file_to_sandbox(
    sandbox_id: str, local_file_path: str, sandbox_file_path: str = "/home/user"
) -> str:
    """Upload a local file to the `/home/user` dir of the remote python interpreter.

    Args:
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.
        local_file_path: The path of the file on local machine to upload.
        sandbox_file_path: The path of directory to upload the file to in the sandbox. Default is `/home/user/`.

    Returns:
        The path of the uploaded file in the remote python interpreter if the upload is successful.
    """
    try:
        sandbox = Sandbox.connect(sandbox_id, api_key=E2B_API_KEY)
    except Exception:
        return f"Failed to connect to sandbox {sandbox_id}"

    sandbox.set_timeout(
        DEFAULT_TIMEOUT
    )  # refresh the timeout for each command execution

    # Get the uploaded file path
    uploaded_file_path = os.path.join(
        sandbox_file_path, os.path.basename(local_file_path)
    )

    # Upload the file
    try:
        with open(local_file_path, "rb") as f:
            sandbox.files.write(uploaded_file_path, f)
    except Exception as e:
        return f"Failed to upload file {local_file_path} to sandbox {sandbox_id}: {e}"

    return f"File uploaded to {uploaded_file_path}"


@mcp.tool()
async def download_internet_file_to_sandbox(
    sandbox_id: str, url: str, sandbox_file_path: str = "/home/user"
) -> str:
    """Download a file from the internet to the `/home/user` dir of the remote python interpreter.

    Args:
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.
        url: The URL of the file to download.
        sandbox_file_path: The path of directory to download the file to in the sandbox. Default is `/home/user/`.

    Returns:
        The path of the downloaded file in the python interpreter if the download is successful.
    """
    try:
        sandbox = Sandbox.connect(sandbox_id, api_key=E2B_API_KEY)
    except Exception:
        return f"Failed to connect to sandbox {sandbox_id}"

    sandbox.set_timeout(
        DEFAULT_TIMEOUT
    )  # refresh the timeout for each command execution

    downloaded_file_path = os.path.join(sandbox_file_path, os.path.basename(url))

    # Download the file
    result = sandbox.commands.run(f"wget {url} -O {downloaded_file_path}")
    if result.exit_code == 0:
        return f"File downloaded to {downloaded_file_path}"
    else:
        return f"Failed to download file from {url} to {downloaded_file_path}: {result}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
