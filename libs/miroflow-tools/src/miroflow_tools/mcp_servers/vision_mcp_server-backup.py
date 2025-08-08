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

import base64
import os

from anthropic import Anthropic
from openai import OpenAI
from fastmcp import FastMCP

# Anthropic credentials
ENABLE_CLAUDE_VISION = os.environ.get("ENABLE_CLAUDE_VISION", "false").lower() == "true"
ENABLE_OPENAI_VISION = os.environ.get("ENABLE_OPENAI_VISION", "false").lower() == "true"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


# Initialize FastMCP server
mcp = FastMCP("vision-mcp-server")


async def guess_mime_media_type_from_extension(file_path: str) -> str:
    """Guess the MIME type based on the file extension."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".gif":
        return "image/gif"
    elif ext == ".webp":
        return "image/webp"
    else:
        return "image/jpeg"  # Default to JPEG if unknown


async def call_claude_vision(image_path_or_url: str, question: str) -> str:
    """Call Claude vision API."""
    messages_for_llm = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": None,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    try:
        if os.path.exists(image_path_or_url):  # Check if the file exists locally
            with open(image_path_or_url, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                messages_for_llm[0]["content"][0]["source"] = dict(
                    type="base64",
                    media_type=await guess_mime_media_type_from_extension(
                        image_path_or_url
                    ),
                    data=image_data,
                )
        else:  # Otherwise, assume it's a URL
            messages_for_llm[0]["content"][0]["source"] = dict(
                type="url", url=image_path_or_url
            )

        client = Anthropic(
            api_key=ANTHROPIC_API_KEY,
            base_url=ANTHROPIC_BASE_URL,
        )

        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            messages=messages_for_llm,
        )

        return response.content[0].text

    except Exception as e:
        return f"Claude Error: {e}"


async def call_openai_vision(image_path_or_url: str, question: str) -> str:
    """Call OpenAI vision API."""
    try:
        if os.path.exists(image_path_or_url):  # Check if the file exists locally
            with open(image_path_or_url, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                mime_type = await guess_mime_media_type_from_extension(
                    image_path_or_url
                )
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                }
        else:  # Otherwise, assume it's a URL
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_path_or_url},
            }

        messages_for_llm = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    image_content,
                ],
            }
        ]

        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=messages_for_llm,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"OpenAI Error: {e}"


@mcp.tool()
async def visual_question_answering(image_path_or_url: str, question: str) -> str:
    """This tool is used to ask question about an image or a video and get the answer with both Claude and OpenAI vision language models. It also automatically performs OCR (text extraction) on the image for additional context.

    Args:
        image_path_or_url: The path of the image file locally or its URL.
        question: The question to ask about the image.

    Returns:
        The concatenated answers from both Claude and OpenAI vision models, including both VQA responses and OCR results.
    """

    # OCR-specific prompt
    ocr_prompt = "Please extract all text from this image. Return only the text content, maintaining the original formatting and structure as much as possible. If there is no text in the image, respond with 'No text found'."

    # Call both APIs for both VQA and OCR
    combined_results = []

    if ENABLE_CLAUDE_VISION:
        if not ANTHROPIC_API_KEY:
            return "Error: claude is enabled but ANTHROPIC_API_KEY is not set."

        # Get OCR result
        claude_ocr_result = await call_claude_vision(image_path_or_url, ocr_prompt)
        combined_results.append(f"**Claude OCR:**\n{claude_ocr_result}")

        # Get VQA result
        claude_vqa_result = await call_claude_vision(image_path_or_url, question)
        combined_results.append(f"**Claude VQA:**\n{claude_vqa_result}")

    if ENABLE_OPENAI_VISION:
        if not OPENAI_API_KEY:
            return "Error: openai is enabled but OPENAI_API_KEY is not set."

        # Get OCR result
        openai_ocr_result = await call_openai_vision(image_path_or_url, ocr_prompt)
        combined_results.append(f"**OpenAI OCR:**\n{openai_ocr_result}")

        # Get VQA result
        openai_vqa_result = await call_openai_vision(image_path_or_url, question)
        combined_results.append(f"**OpenAI VQA:**\n{openai_vqa_result}")

    if not combined_results:
        return "Error: No responses received from either APIs or key."

    return "\n\n".join(combined_results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
