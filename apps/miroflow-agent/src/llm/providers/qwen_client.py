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


import dataclasses
import logging
import os
from typing import Any, Dict, List
import tiktoken
import asyncio

from openai import AsyncOpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient, OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from ..provider_client_base import LLMProviderClientBase
from ...utils.prompt_utils import generate_mcp_system_prompt

logger = logging.getLogger()


@dataclasses.dataclass
class QwenLLMClient(LLMProviderClientBase):
    def _create_client(self):
        """Create configured Qwen client"""

        QWEN_API_KEY = os.environ.get("QWEN_API_KEY", None)

        http_client_args = {}
        if os.environ.get("HTTPS_PROXY"):
            http_client_args["proxy"] = os.environ.get("HTTPS_PROXY")
            logger.info(f"Info: Using proxy {http_client_args['proxy']}")

        if self.async_client:
            return AsyncOpenAI(
                api_key=QWEN_API_KEY,
                base_url=self.openai_base_url,
                http_client=DefaultAsyncHttpxClient(**http_client_args),
            )
        else:
            return OpenAI(
                api_key=QWEN_API_KEY,
                base_url=self.openai_base_url,
                http_client=DefaultHttpxClient(**http_client_args),
            )

    def _update_token_usage(self, usage_data):
        if usage_data:
            input_tokens = getattr(usage_data, "prompt_tokens", 0)
            output_tokens = getattr(usage_data, "completion_tokens", 0)
            prompt_tokens_details = getattr(usage_data, "prompt_tokens_details", None)
            if prompt_tokens_details:
                cached_tokens = getattr(prompt_tokens_details, "cached_tokens", 0)
            else:
                cached_tokens = 0

            # Record token usage for the most recent call
            self.last_call_tokens = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            }

            # OpenAI does not provide cache_creation_input_tokens
            self.token_usage["total_input_tokens"] += input_tokens
            self.token_usage["total_output_tokens"] += output_tokens
            self.token_usage["total_cache_read_input_tokens"] += cached_tokens

            logger.info(
                f"Current round token usage - Input: {self.token_usage['total_input_tokens']}, "
                f"Output: {self.token_usage['total_output_tokens']}"
            )

    @retry(wait=wait_fixed(30), stop=stop_after_attempt(10))
    async def _create_message(
        self,
        system_prompt: str,
        messages_history: List[Dict[str, Any]],
        tools_definitions,
        keep_tool_result: int = -1,
    ):
        """
        Send message to OpenAI API.
        :param system_prompt: System prompt string.
        :param messages: Message history list.
        :return: OpenAI API response object or None (if error occurs).
        """

        # put the system prompt in the first message since OpenAI API does not support system prompt in
        if system_prompt:
            # Check if there's already a system or developer message
            if messages_history and messages_history[0]["role"] in [
                "system",
                "developer",
            ]:
                messages_history[0] = {
                    "role": "system",
                    "content": system_prompt,
                }

            else:
                messages_history.insert(
                    0,
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                )

        messages_history = self._remove_tool_result_from_messages(
            messages_history, keep_tool_result
        )

        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages_history,
            "tools": [],
            "stream": False,
            "top_p": self.top_p,
        }

        try:
            if self.async_client:
                response = await self.client.chat.completions.create(**params)
            else:
                response = self.client.chat.completions.create(**params)
            # Update token count
            self._update_token_usage(getattr(response, "usage", None))
            self.task_log.log_step(
                "Qwen LLM | Response Status",
                f"LLM call status: {getattr(response.choices[0], 'finish_reason', 'N/A')}",
                "info",
            )

            return response, messages_history

        except asyncio.TimeoutError as e:
            self.task_log.log_step(
                "Qwen LLM | Timeout Error",
                f"Timeout error: {str(e)}",
                "error",
            )
            raise e
        except asyncio.CancelledError as e:
            self.task_log.log_step(
                "Qwen LLM | Request Cancelled",
                f"Request was cancelled: {str(e)}",
                "error",
            )
            raise e
        except Exception as e:
            if "Error code: 400" in str(e) and "longer than the model" in str(e):
                self.task_log.log_step(
                    "Qwen LLM | Context Length Error",
                    f"Error: {str(e)}",
                    "fatal error, exceed max context length, this should not happen as the context length is checked before the call",
                )
                raise e
            else:
                self.task_log.log_step(
                    "Qwen LLM | API Error",
                    f"Error: {str(e)}",
                    "error",
                )
                raise e

    def process_llm_response(
        self, llm_response, message_history, agent_type="main"
    ) -> tuple[str, bool, list]:
        """Process OpenAI LLM response"""

        if not llm_response or not llm_response.choices:
            error_msg = "LLM did not return a valid response."
            logger.info(f"Error: {error_msg}")
            return "", True, message_history  # Exit loop, return message_history

        # Extract LLM response text
        if llm_response.choices[0].finish_reason == "stop":
            assistant_response_text = llm_response.choices[0].message.content or ""

            message_history.append(
                {"role": "assistant", "content": assistant_response_text}
            )

        elif llm_response.choices[0].finish_reason == "length":
            assistant_response_text = llm_response.choices[0].message.content or ""
            if assistant_response_text == "":
                assistant_response_text = "LLM response is empty."
            elif "Context length exceeded" in assistant_response_text:
                # This is the case where context length is exceeded, needs special handling
                logger.warning(
                    "Detected context length exceeded, returning error status"
                )
                message_history.append(
                    {"role": "assistant", "content": assistant_response_text}
                )
                return (
                    assistant_response_text,
                    True,
                    message_history,
                )  # Return True to indicate need to exit loop

            message_history.append(
                {"role": "assistant", "content": assistant_response_text}
            )

        else:
            # Different from Openai Client, we don't use tool calls for qwen,
            # so we don't support tool_call finish reason
            raise ValueError(
                f"Unsupported finish reason: {llm_response.choices[0].finish_reason}"
            )

        return assistant_response_text, False, message_history

    def extract_tool_calls_info(self, llm_response, assistant_response_text) -> list:
        """Extract tool call information from Qwen LLM response"""
        from ...utils.parsing_utils import parse_llm_response_for_tool_calls

        # For qwen, use the same parsing method as anthropic
        return parse_llm_response_for_tool_calls(assistant_response_text)

    def update_message_history(self, message_history, all_tool_results_content_with_id):
        """Update message history with tool calls data (llm client specific)"""

        merged_text = "\n".join(
            [
                item[1]["text"]
                for item in all_tool_results_content_with_id
                if item[1]["type"] == "text"
            ]
        )

        message_history.append(
            {
                "role": "user",
                "content": merged_text,
            }
        )

        return message_history

    def generate_agent_system_prompt(self, date, mcp_servers) -> str:
        return generate_mcp_system_prompt(date, mcp_servers)

    def _estimate_tokens(self, text: str) -> int:
        """Use tiktoken to estimate the number of tokens in text"""
        if not hasattr(self, "encoding"):
            # Initialize tiktoken encoder
            try:
                self.encoding = tiktoken.get_encoding("o200k_base")
            except Exception:
                # If o200k_base is not available, use cl100k_base as fallback
                self.encoding = tiktoken.get_encoding("cl100k_base")

        try:
            return len(self.encoding.encode(text))
        except Exception:
            # If encoding fails, use simple estimation: approximately 1 token per 4 characters
            self.task_log.log_step(
                "Qwen LLM | Token Estimation Error",
                f"Error: {str(e)}",
                "error",
            )
            return len(text) // 4

    def ensure_summary_context(
        self, message_history: list, summary_prompt: str
    ) -> bool:
        """
        Check if current message_history + summary_prompt will exceed context
        If it will exceed, remove the last assistant-user pair and return False
        Return True to continue, False if messages have been rolled back
        """
        # Get token usage from the last LLM call
        last_prompt_tokens = self.last_call_tokens.get("prompt_tokens", 0)
        last_completion_tokens = self.last_call_tokens.get("completion_tokens", 0)
        buffer_factor = 2

        # Calculate token count for summary prompt
        summary_tokens = self._estimate_tokens(summary_prompt) * buffer_factor

        # Calculate token count for the last user message in message_history (if exists and not sent)
        last_user_tokens = 0
        if message_history[-1]["role"] == "user":
            content = message_history[-1]["content"]
            last_user_tokens = self._estimate_tokens(content) * buffer_factor

        # Calculate total token count: last prompt + completion + last user message + summary + reserved response space
        estimated_total = (
            last_prompt_tokens
            + last_completion_tokens
            + last_user_tokens
            + summary_tokens
            + self.max_tokens
            + 1000  # Add 1000 tokens as buffer
        )

        if estimated_total >= self.max_context_length:
            self.task_log.log_step(
                "Qwen LLM | Context Limit Reached",
                "Context limit reached, proceeding to step back and summarize the conversation",
                "info",
            )

            # Remove the last user message (tool call results)
            if message_history[-1]["role"] == "user":
                message_history.pop()

            # Remove the second-to-last assistant message (tool call request)
            if message_history[-1]["role"] == "assistant":
                message_history.pop()

            self.task_log.log_step(
                "Qwen LLM | Context Limit Reached",
                f"Removed the last assistant-user pair, current message_history length: {len(message_history)}",
                "info",
            )

            return False, message_history

        self.task_log.log_step(
            "Qwen LLM | Context Limit Not Reached",
            f"Context limit not reached, proceeding to continue, current context length: {estimated_total}/{self.max_context_length}",
            "info",
        )
        return True, message_history

    def handle_max_turns_reached_summary_prompt(self, message_history, summary_prompt):
        """Handle max turns reached summary prompt"""
        if message_history[-1]["role"] == "user":
            message_history.pop()  # Remove the last user message
            # TODO: this part is a temporary fix, we need to find a better way to handle this
            return summary_prompt
            # return (
            #     last_user_message["content"]
            #     + "\n*************\n"
            #     + summary_prompt
            # )
        else:
            return summary_prompt

    def format_token_usage_summary(self):
        """Format token usage statistics and cost estimation, return summary_lines for format_final_summary and log string - Qwen implementation"""
        token_usage = self.get_token_usage()

        total_input = token_usage.get("total_input_tokens", 0)
        total_output = token_usage.get("total_output_tokens", 0)
        cache_input = token_usage.get("total_cache_input_tokens", 0)

        # Actual cost (considering cache)
        cost = (
            ((total_input - cache_input) / 1_000_000 * self.input_token_price)
            + (cache_input / 1_000_000 * self.cache_input_token_price)
            + (total_output / 1_000_000 * self.output_token_price)
        )

        summary_lines = []
        summary_lines.append("\n" + "-" * 20 + " Token Usage & Cost " + "-" * 20)
        summary_lines.append(f"Total Input Tokens: {total_input}")
        summary_lines.append(f"Total Cache Input Tokens: {cache_input}")
        summary_lines.append(f"Total Output Tokens: {total_output}")
        summary_lines.append("-" * (40 + len(" Token Usage & Cost ")))
        summary_lines.append(f"Input Token Price: ${self.input_token_price:.4f} USD")
        summary_lines.append(f"Output Token Price: ${self.output_token_price:.4f} USD")
        summary_lines.append(
            f"Cache Input Token Price: ${self.cache_input_token_price:.4f} USD"
        )
        summary_lines.append("-" * (40 + len(" Token Usage & Cost ")))
        summary_lines.append(f"Estimated Cost (with cache): ${cost:.4f} USD")
        summary_lines.append("-" * (40 + len(" Token Usage & Cost ")))

        # Generate log string
        log_string = f"[Qwen/{self.model_name}] Total Input: {total_input}, Cache Input: {cache_input}, Output: {total_output}, Input Price: ${self.input_token_price:.4f} USD, Cache Input Price: ${self.cache_input_token_price:.4f} USD, Output Price: ${self.output_token_price:.4f} USD, Cost: ${cost:.4f} USD"

        return summary_lines, log_string

    def get_token_usage(self):
        return self.token_usage.copy()
