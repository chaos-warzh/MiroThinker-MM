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

import time
from datetime import date
from typing import Dict, Optional, Tuple

from miroflow_tools.manager import ToolManager
from miroflow_tracing import function_span
from omegaconf import DictConfig

from ..config.settings import expose_sub_agents_as_tools
from ..io.input_handler import process_input
from ..io.output_formatter import OutputFormatter
from ..llm.client import LLMClient
from ..logging.task_logger import (
    TaskLog,
    get_utc_plus_8_time,
    logger,
)
from ..utils.prompt_utils import (
    generate_agent_specific_system_prompt,
    generate_agent_summarize_prompt,
)


def _list_tools(sub_agent_tool_managers: Dict[str, ToolManager]):
    # Use a dictionary to store the cached result
    cache = None

    async def wrapped():
        nonlocal cache
        if cache is None:
            # Only fetch tool definitions if not already cached
            result = {
                name: await tool_manager.get_all_tool_definitions()
                for name, tool_manager in sub_agent_tool_managers.items()
            }
            cache = result
        return cache

    return wrapped


class Orchestrator:
    def __init__(
        self,
        main_agent_tool_manager: ToolManager,
        sub_agent_tool_managers: Dict[str, ToolManager],
        llm_client: LLMClient,
        output_formatter: OutputFormatter,
        cfg: DictConfig,
        task_log: Optional["TaskLog"] = None,
    ):
        self.main_agent_tool_manager = main_agent_tool_manager
        self.sub_agent_tool_managers = sub_agent_tool_managers
        self.llm_client = llm_client
        self.output_formatter = output_formatter
        self.cfg = cfg
        self.task_log = task_log
        # call this once, then use cache value
        self._list_sub_agent_tools = _list_tools(sub_agent_tool_managers)

        # Pass task_log to llm_client
        if self.llm_client and task_log:
            self.llm_client.task_log = task_log

    async def _handle_llm_call(
        self,
        system_prompt,
        message_history,
        tool_definitions,
        step_id: int,
        purpose: str = "",
        keep_tool_result: int = -1,
        agent_type: str = "main",
    ) -> Tuple[Optional[str], bool, Optional[object]]:
        """Unified LLM call and logging processing
        Returns:
            Tuple[Optional[str], bool, Optional[object]]: (response_text, should_break, tool_calls_info)
        """

        try:
            response, message_history = await self.llm_client.create_message(
                system_prompt=system_prompt,
                message_history=message_history,
                tool_definitions=tool_definitions,
                keep_tool_result=self.cfg.agent.keep_tool_result,
                step_id=step_id,
                task_log=self.task_log,
                agent_type=agent_type,
            )

            # Check if response is None (indicating an error occurred)
            if response is None:
                self.task_log.log_step(
                    f"{purpose} | LLM Call Failed",
                    f"{purpose} failed - no response received",
                    "Error",
                )
                return "", True, None, message_history

            # Use client's response processing method
            assistant_response_text, should_break, message_history = (
                self.llm_client.process_llm_response(
                    response, message_history, agent_type
                )
            )

            # Use client's tool call information extraction method
            tool_calls_info = self.llm_client.extract_tool_calls_info(
                response, assistant_response_text
            )

            self.task_log.log_step(
                f"{purpose} | LLM Call",
                f"{purpose} completed successfully",
            )
            return (
                assistant_response_text,
                should_break,
                tool_calls_info,
                message_history,
            )

        except Exception as e:
            self.task_log.log_step(
                f"{purpose} | LLM Call ERROR",
                f"{purpose} error: {str(e)}",
                "Error",
            )
            # Return empty response with should_break=True to indicate error
            return "", True, None, message_history

    async def run_sub_agent(
        self, sub_agent_name, task_description, keep_tool_result: int = -1
    ):
        """
        Run sub agent
        """
        self.task_log.log_step(
            "Sub Agent | Start Task", f"Starting sub agent {sub_agent_name}"
        )
        task_description += "\n\nPlease provide the answer and detailed supporting information of the subtask given to you."
        self.task_log.log_step(
            "Sub Agent | Task Description", f"Subtask: {task_description}"
        )

        # Start new sub-agent session
        self.task_log.start_sub_agent_session(sub_agent_name, task_description)

        # Simplified initial user content (no file attachments)
        initial_user_content = task_description
        message_history = [{"role": "user", "content": initial_user_content}]

        # Get sub-agent tool definitions
        tool_definitions = await self._list_sub_agent_tools()
        tool_definitions = tool_definitions.get(sub_agent_name, {})
        self.task_log.log_step(
            "Sub Agent | Get Tool Definitions",
            f"Number of tools: {len(tool_definitions)}",
        )

        if not tool_definitions:
            self.task_log.log_step(
                "Sub Agent | No Tools",
                f"No tool definitions available for {sub_agent_name}",
            )

        # Generate sub-agent system prompt
        system_prompt = self.llm_client.generate_agent_system_prompt(
            date=date.today(),
            mcp_servers=tool_definitions,
        ) + generate_agent_specific_system_prompt(agent_type=sub_agent_name)

        # Limit sub-agent turns
        max_turns = self.cfg.agent.sub_agents[sub_agent_name].max_turns
        turn_count = 0
        all_tool_results_content_with_id = []

        while turn_count < max_turns:
            turn_count += 1
            self.task_log.log_step(
                f"Sub Agent | Turn: {turn_count}", f"Starting turn {turn_count}"
            )
            self.task_log.save()

            # Reset 'last_call_tokens'
            self.llm_client.last_call_tokens = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

            # Use unified LLM call processing
            (
                assistant_response_text,
                should_break,
                tool_calls,
                message_history,
            ) = await self._handle_llm_call(
                system_prompt,
                message_history,
                tool_definitions,
                turn_count,
                f"Sub Agent: {sub_agent_name} | Turn: {turn_count}",
                keep_tool_result=keep_tool_result,
                agent_type=sub_agent_name,
            )

            if should_break:
                self.task_log.log_step(
                    f"Sub Agent | Turn: {turn_count} | LLM Call",
                    "should break is True, breaking the loop",
                    "info",
                )
                break

            # Process LLM response
            elif assistant_response_text:
                self.task_log.log_step(
                    f"Sub Agent | Turn: {turn_count} | LLM Call",
                    "LLM call completed successfully",
                    "info",
                )

            else:
                # LLM call failed, end current turn
                self.task_log.log_step(
                    f"Sub Agent | Turn: {turn_count} | LLM Call",
                    "LLM call failed",
                    "info",
                )
                break

            # Use tool calls parsed from LLM response
            if not tool_calls:
                self.task_log.log_step(
                    f"Sub Agent | Turn: {turn_count} | LLM Call",
                    f"No tool calls found in sub agent {sub_agent_name}, ending on turn {turn_count}",
                    "info",
                )
                break

            # Execute tool calls
            tool_calls_data = []
            all_tool_results_content_with_id = []

            for call in tool_calls:
                server_name = call["server_name"]
                tool_name = call["tool_name"]
                arguments = call["arguments"]
                call_id = call["id"]

                self.task_log.log_step(
                    f"Sub Agent | Turn: {turn_count} | Tool Call",
                    f"Executing {tool_name} on {server_name}",
                    "info",
                )

                call_start_time = time.time()
                try:
                    with function_span(
                        name=f"{server_name}.{tool_name}", input=arguments
                    ) as span:
                        tool_result = await self.sub_agent_tool_managers[
                            sub_agent_name
                        ].execute_tool_call(server_name, tool_name, arguments)
                        span.span_data.output = str(tool_result)

                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    self.task_log.log_step(
                        f"Sub Agent | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} executed successfully in {call_duration_ms}ms",
                        "info",
                    )

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "result": tool_result,
                            "duration_ms": call_duration_ms,
                            "call_time": get_utc_plus_8_time(),
                        }
                    )

                except Exception as e:
                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "error": str(e),
                            "duration_ms": call_duration_ms,
                            "call_time": get_utc_plus_8_time(),
                        }
                    )
                    tool_result = {
                        "error": f"Tool call failed: {str(e)}",
                        "server_name": server_name,
                        "tool_name": tool_name,
                    }
                    self.task_log.log_step(
                        f"Sub Agent | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} failed to execute: {str(e)}",
                        "info",
                    )

                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                    tool_result
                )

                all_tool_results_content_with_id.append((call_id, tool_result_for_llm))

            # Record tool calls to current sub-agent turn
            message_history = self.llm_client.update_message_history(
                message_history, all_tool_results_content_with_id
            )

            # Generate summary_prompt to check token limits
            temp_summary_prompt = generate_agent_summarize_prompt(
                task_description,
                task_failed=True,  # Temporarily set to True to simulate task failure
                agent_type=sub_agent_name,
            )

            pass_length_check, message_history = self.llm_client.ensure_summary_context(
                message_history, temp_summary_prompt
            )

            # Check if current context will exceed limits, if so automatically rollback messages and trigger summary
            if not pass_length_check:
                # Context exceeded limits, set turn_count to trigger summary
                turn_count = max_turns
                self.task_log.log_step(
                    f"Sub Agent | Turn: {turn_count} | Context Limit Reached",
                    "Context limit reached, triggering summary",
                    "info",
                )
                break

        # Continue processing
        self.task_log.log_step(
            "Sub Agent | Main Loop Completed",
            f"Main loop completed after {turn_count} turns",
        )

        # Record browsing agent loop end
        if turn_count >= max_turns:
            self.task_log.log_step(
                "Sub Agent | Max Turns Reached / Context Limit Reached",
                f"Reached maximum turns ({max_turns}) or context limit reached",
            )

        else:
            self.task_log.log_step(
                "Sub Agent | Main Loop Completed",
                f"Main loop completed after {turn_count} turns",
            )

        # Final summary
        self.task_log.log_step(
            "Sub Agent | Final Summary",
            f"Generating sub agent {sub_agent_name} final summary",
        )

        # Generate sub agent summary prompt
        summary_prompt = generate_agent_summarize_prompt(
            task_description,
            task_failed=(turn_count >= max_turns),
            agent_type=sub_agent_name,
        )

        # If maximum turns reached, output previous turn tool call results and start summary system prompt
        if turn_count >= max_turns:
            summary_prompt = self.llm_client.handle_max_turns_reached_summary_prompt(
                message_history, summary_prompt
            )

        message_history.append({"role": "user", "content": summary_prompt})

        # Use unified LLM call processing to generate final summary
        (
            final_answer_text,
            should_break,
            tool_calls_info,
            message_history,
        ) = await self._handle_llm_call(
            system_prompt,
            message_history,
            tool_definitions,
            turn_count + 1,
            f"Sub agent: {sub_agent_name} | Final summary",
            keep_tool_result=keep_tool_result,
            agent_type=sub_agent_name,
        )

        if final_answer_text:
            self.task_log.log_step(
                "Sub Agent | Final Answer",
                f"Sub agent {sub_agent_name} final answer generated successfully",
            )

        else:
            final_answer_text = (
                f"No final answer generated by sub agent {sub_agent_name}."
            )
            self.task_log.log_step(
                "Sub Agent | Final Answer",
                f"Unable to generate sub agent {sub_agent_name} final answer",
                "Error",
            )

        self.task_log.log_step(
            "Sub Agent | Final Answer",
            f"Sub agent {sub_agent_name} final answer: {final_answer_text}",
        )

        self.task_log.sub_agent_message_history_sessions[
            self.task_log.current_sub_agent_session_id
        ] = {"system_prompt": system_prompt, "message_history": message_history}

        self.task_log.save()

        self.task_log.end_sub_agent_session(sub_agent_name)
        self.task_log.log_step(
            "Sub Agent | Completed", f"Sub agent {sub_agent_name} completed"
        )

        # Return final answer instead of conversation log, so main agent can use it directly
        return final_answer_text

    async def run_main_agent(
        self, task_description, task_file_name=None, task_id="default_task"
    ):
        """
        Execute the main end-to-end task.
        """
        keep_tool_result = int(self.cfg.agent.keep_tool_result)

        self.task_log.log_step("Main Agent | Start Task", f"Task ID: {task_id}")
        self.task_log.log_step(
            "Main Agent | Task Description", f"Task Description: {task_description}"
        )
        if task_file_name:
            self.task_log.log_step(
                "Main Agent | Associated File", f"Associated File: {task_file_name}"
            )

        # 1. Process input
        initial_user_content, processed_task_desc = process_input(
            task_description, task_file_name
        )
        message_history = [{"role": "user", "content": initial_user_content}]

        # Record initial user input
        user_input = processed_task_desc
        if task_file_name:
            user_input += f"\n[Attached file: {task_file_name}]"

        # 2. Get tool definitions
        tool_definitions = await self.main_agent_tool_manager.get_all_tool_definitions()
        tool_definitions += expose_sub_agents_as_tools(self.cfg.agent.sub_agents)
        if not tool_definitions:
            logger.info(
                "Warning: No tool definitions found. LLM cannot use any tools."
            )

        self.task_log.log_step(
            "Main Agent | Get Tool Definitions",
            f"Number of tools: {len(tool_definitions)}",
        )

        # 3. Generate system prompt
        system_prompt = self.llm_client.generate_agent_system_prompt(
            date=date.today(),
            mcp_servers=tool_definitions,
        ) + generate_agent_specific_system_prompt(agent_type="main")

        # 4. Main loop: LLM <-> Tools
        max_turns = self.cfg.agent.main_agent.max_turns
        turn_count = 0
        error_msg = ""
        tool_calls = []
        while turn_count < max_turns:
            turn_count += 1
            logger.info(f"\n--- Main Agent Turn {turn_count} ---")
            self.task_log.save()

            # Use unified LLM call processing
            (
                assistant_response_text,
                should_break,
                tool_calls,
                message_history,
            ) = await self._handle_llm_call(
                system_prompt,
                message_history,
                tool_definitions,
                turn_count,
                f"Main agent | Turn: {turn_count}",
                keep_tool_result=keep_tool_result,
                agent_type="main",
            )

            # Process LLM response
            if assistant_response_text:
                if should_break:
                    self.task_log.log_step(
                        f"Main Agent | Turn: {turn_count} | LLM Call",
                        "should break is True, breaking the loop",
                        "info",
                    )
                    break

            else:
                self.task_log.log_step(
                    f"Main Agent | Turn: {turn_count} | LLM Call",
                    "No valid response from LLM, breaking the loop",
                    "info",
                )
                break

            if not tool_calls:
                self.task_log.log_step(
                    f"Main Agent | Turn: {turn_count} | LLM Call",
                    "LLM did not request tool usage, ending process.",
                    "info",
                )
                break

            # 7. Execute tool calls (execute in order)
            tool_calls_data = []
            all_tool_results_content_with_id = []

            self.task_log.log_step(
                f"Main Agent | Turn: {turn_count} | Tool Calls",
                f"Number of tool calls detected: {len(tool_calls)}",
                "info",
            )

            main_agent_last_call_tokens = self.llm_client.last_call_tokens

            for call in tool_calls:
                server_name = call["server_name"]
                tool_name = call["tool_name"]
                arguments = call["arguments"]
                call_id = call["id"]

                call_start_time = time.time()
                try:
                    if server_name.startswith("agent-"):
                        with function_span(
                            name=f"{server_name}.{tool_name}", input=arguments
                        ) as span:
                            sub_agent_result = await self.run_sub_agent(
                                server_name, arguments["subtask"], keep_tool_result
                            )
                            tool_result = {
                                "server_name": server_name,
                                "tool_name": tool_name,
                                "result": sub_agent_result,
                            }
                            span.span_data.output = str(tool_result)
                    else:
                        with function_span(
                            name=f"{server_name}.{tool_name}", input=arguments
                        ) as span:
                            tool_result = (
                                await self.main_agent_tool_manager.execute_tool_call(
                                    server_name=server_name,
                                    tool_name=tool_name,
                                    arguments=arguments,
                                )
                            )
                            span.span_data.output = str(tool_result)

                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "result": tool_result,
                            "duration_ms": call_duration_ms,
                            "call_time": get_utc_plus_8_time(),
                        }
                    )
                    self.task_log.log_step(
                        f"Main Agent | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} executed successfully in {call_duration_ms}ms",
                        "info",
                    )

                except Exception as e:
                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "error": str(e),
                            "duration_ms": call_duration_ms,
                            "call_time": get_utc_plus_8_time(),
                        }
                    )
                    tool_result = {
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "error": str(e),
                    }
                    self.task_log.log_step(
                        f"Main Agent | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} failed to execute: {str(e)}",
                        "info",
                    )

                # Format results to feedback to LLM (more concise)
                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                    tool_result
                )
                all_tool_results_content_with_id.append((call_id, tool_result_for_llm))

            # Update 'last_call_tokens'
            self.llm_client.last_call_tokens = main_agent_last_call_tokens

            # Update message history with tool calls data (llm client specific)
            message_history = self.llm_client.update_message_history(
                message_history, all_tool_results_content_with_id
            )

            self.task_log.main_agent_message_history = {
                "system_prompt": system_prompt,
                "message_history": message_history,
            }
            self.task_log.save()

            # Assess current context length, determine if we need to trigger summary
            temp_summary_prompt = generate_agent_summarize_prompt(
                task_description,
                task_failed=True,  # Temporarily set to True to simulate task failure
                agent_type="main",
            )

            pass_length_check, message_history = self.llm_client.ensure_summary_context(
                message_history, temp_summary_prompt
            )

            # Check if current context will exceed limits, if so automatically rollback messages and trigger summary
            if not pass_length_check:
                turn_count = max_turns
                self.task_log.log_step(
                    f"Main Agent | Turn: {turn_count} | Context Limit Reached",
                    "Context limit reached, triggering summary",
                    "warning",
                )
                break

        # Record main loop end
        if turn_count >= max_turns:
            self.task_log.log_step(
                "Main Agent | Max Turns Reached / Context Limit Reached",
                f"Reached maximum turns ({max_turns}) or context limit reached",
                "warning",
            )

        else:
            self.task_log.log_step(
                "Main Agent | Main Loop Completed",
                f"Main loop completed after {turn_count} turns",
            )

        # Final summary
        self.task_log.log_step("Main Agent | Final Summary", "Generating final summary")

        # Generate summary prompt (generate only once)
        summary_prompt = generate_agent_summarize_prompt(
            task_description,
            task_failed=(error_msg != "") or (turn_count >= max_turns),
            agent_type="main",
        )

        # If maximum turns reached, output previous turn tool call results and start summary system prompt
        if turn_count >= max_turns:
            summary_prompt = self.llm_client.handle_max_turns_reached_summary_prompt(
                message_history, summary_prompt
            )

        message_history.append({"role": "user", "content": summary_prompt})

        # Use unified LLM call processing
        (
            final_answer_text,
            should_break,
            tool_calls_info,
            message_history,
        ) = await self._handle_llm_call(
            system_prompt,
            message_history,
            tool_definitions,
            turn_count + 1,
            "Main agent | Final Summary",
            keep_tool_result=keep_tool_result,
            agent_type="main",
        )

        self.task_log.main_agent_message_history = {
            "system_prompt": system_prompt,
            "message_history": message_history,
        }
        self.task_log.save()

        # Process response results
        if final_answer_text:
            self.task_log.log_step(
                "Main Agent | Final Answer", "Final answer extracted successfully"
            )

            # Log the final answer
            self.task_log.log_step(
                "Main Agent | Final Answer",
                f"Final answer content: {final_answer_text}",
            )

        else:
            final_answer_text = "No final answer generated."
            self.task_log.log_step(
                "Main Agent | Final Answer",
                "Unable to extract final answer",
                "Error",
            )

        final_summary, final_boxed_answer, usage_log = (
            self.output_formatter.format_final_summary_and_log(
                final_answer_text, self.llm_client
            )
        )

        self.task_log.log_step(
            "Main Agent | Final boxed answer",
            f"Final boxed answer: {final_boxed_answer}",
        )

        self.task_log.log_step(
            "Main Agent | Usage Calculation", f"Usage log: {usage_log}"
        )

        self.task_log.log_step(
            "Main Agent | Task Completed",
            f"Main agent task {task_id} completed successfully",
        )

        return final_summary, final_boxed_answer
