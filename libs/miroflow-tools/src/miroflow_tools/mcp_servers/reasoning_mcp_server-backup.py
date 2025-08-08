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

from fastmcp import FastMCP
from openai import OpenAI

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")

mcp = FastMCP("reasoning-mcp-server")


@mcp.tool()
async def reasoning(question: str) -> str:
    """You can use this tool use solve hard math problem, puzzle, riddle and IQ test question that requries a lot of chain of thought efforts.
    DO NOT use this tool for simple and obvious question.

    Args:
        question: The hard question.

    Returns:
        The answer to the question.
    """

    messages_for_llm = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question,
                }
            ],
        }
    ]

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    response = client.chat.completions.create(
        model="claude-3-7-sonnet-20250219-thinking",
        max_tokens=21000,
        messages=messages_for_llm,
    )

    try:
        return response.choices[0].message.content
    except Exception:
        # Handle the case where the text just missing
        print("Reasoning Error: only thinking content is returned")
        return response.choices[0].message.reasoning_content



if __name__ == "__main__":
    mcp.run(transport="stdio")
