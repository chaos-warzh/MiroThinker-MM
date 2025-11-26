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
import logging
import os
import time
import uuid
from collections import defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from miroflow_tools.manager import ToolManager
from omegaconf import DictConfig

from ..config.settings import expose_sub_agents_as_tools
from ..io.input_handler import process_input
from ..io.output_formatter import OutputFormatter
from ..llm.factory import ClientFactory
from ..logging.task_logger import (
    TaskLog,
    get_utc_plus_8_time,
)
from ..utils.parsing_utils import extract_llm_response_text
from ..utils.prompt_utils import (
    generate_agent_specific_system_prompt,
    generate_agent_summarize_prompt,
)
from ..utils.wrapper_utils import ErrorBox, ResponseBox

logger = logging.getLogger(__name__)


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
        llm_client: ClientFactory,
        output_formatter: OutputFormatter,
        cfg: DictConfig,
        task_log: Optional["TaskLog"] = None,
        stream_queue: Optional[Any] = None,
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
        sub_agent_tool_definitions: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
        self.main_agent_tool_manager = main_agent_tool_manager
        self.sub_agent_tool_managers = sub_agent_tool_managers
        self.llm_client = llm_client
        self.output_formatter = output_formatter
        self.cfg = cfg
        self.task_log = task_log
        self.stream_queue = stream_queue
        self.tool_definitions = tool_definitions
        self.sub_agent_tool_definitions = sub_agent_tool_definitions
        # call this once, then use cache value
        self._list_sub_agent_tools = _list_tools(sub_agent_tool_managers)
        self.max_repeat_queries = 3

        # Pass task_log to llm_client
        if self.llm_client and task_log:
            self.llm_client.task_log = task_log

        # Record used subtask / q / Query
        self.used_queries = {}

    # =========================
    # Multi-stage report workflow
    # =========================

    async def run_report_workflow(self, task_description: str) -> str:
        """High-level multi-stage report workflow.

        Stages:
        1) Research: use existing browsing sub-agent/tools to collect notes
        2) Outline: generate structured outline (JSON-like) from research notes
        3) Section writing: generate each section's content based on outline
        4) Polish: global refinement + formatting to final markdown report
        """

        self.task_log.log_step(
            "info", "Report Workflow", "Starting multi-stage report workflow",
        )

        research_notes = await self._run_research_phase(task_description)
        outline = await self._generate_outline_phase(task_description, research_notes)
        draft_sections = await self._write_sections_phase(
            task_description, outline, research_notes
        )
        final_report = await self._polish_phase(
            task_description, draft_sections, research_notes, outline=outline
        )

        self.task_log.log_step(
            "info", "Report Workflow", "Completed multi-stage report workflow",
        )

        # Also log a brief preview to console for debugging (truncated)
        preview = (final_report or "").strip()
        if len(preview) > 500:
            preview = preview[:500] + "... [truncated]"
        logger.info("[Report Workflow] Final report preview:\n%s", preview)

        return final_report

    async def _run_research_phase(self, task_description: str) -> List[str]:
        """Stage 1: use existing sub-agent/tools to collect research notes.

        For now we simply call agent-browsing (if configured) as a research
        helper and treat its final answer as a single research note.
        """

        self.task_log.log_step(
            "info", "Report | Research", "Starting research phase",
        )

        research_notes: List[str] = []

        # Prefer an explicit browsing sub-agent if available
        if "agent-browsing" in self.sub_agent_tool_managers:
            try:
                subtask = (
                    task_description
                    + "\n\nPlease collect background information, key facts, data, and references that are helpful for writing a technical report."
                )
                browsing_result = await self.run_sub_agent(
                    "agent-browsing", subtask
                )
                if isinstance(browsing_result, str) and browsing_result.strip():
                    research_notes.append(browsing_result.strip())
            except Exception as e:
                self.task_log.log_step(
                    "warning",
                    "Report | Research",
                    f"agent-browsing failed, skip research phase: {e}",
                )

        # If no notes collected (or no browsing agent), still return a fallback
        if not research_notes:
            research_notes.append(
                "(No external research collected; write based on task description and your own knowledge.)"
            )

        # Log research notes length / brief preview
        preview = "\n---\n".join(note[:500] for note in research_notes)
        self.task_log.log_step(
            "info",
            "Report | Research",
            f"Finished research phase, collected {len(research_notes)} notes. Preview:\n{preview}",
        )
        return research_notes

    async def _generate_outline_phase(
        self, task_description: str, research_notes: List[str]
    ) -> Dict[str, Any]:
        """Stage 2: generate a structured outline for the report.

        We ask the LLM to output JSON-like text. We keep parsing simple here:
        try json.loads, otherwise wrap as a minimal outline with one section.
        """

        self.task_log.log_step(
            "info", "Report | Outline", "Starting outline phase",
        )

        system_prompt = (
            "You are an expert researcher and technical writer. "
            "Create a comprehensive outline for a research report based on the provided task and research notes. "
            "The structure should follow standard academic or professional report conventions.\n"
            "Guidelines:\n"
            "- **Language**: Detect the language of the task description. The outline (titles and summaries) MUST be in the SAME language (Chinese or English).\n"
            "- **Word Count Planning**: Check if the task specifies a total word count (e.g., '1500-2000 words'). If so, you MUST plan the word count for EACH section to ensure the total meets the requirement. Even if not specified, aim for a comprehensive length (~1500 words) and allocate accordingly.\n"
            "- **Structure**: Each section object MUST include a 'word_count' field specifying the target length. The value must be in the SAME language as the outline (e.g., '~300字' for Chinese, '~300 words' for English).\n"
            "- The outline must include 'Introduction' (or '引言'), 'Conclusion' (or '结论'), and 'References' (or '参考资料') sections, translated appropriately.\n"
            "- Organize the main body into logical sections and subsections based on the research findings.\n"
            "- Do NOT use generic section titles like 'Main Report Body' or 'Body'. Use descriptive titles.\n"
            "- Output STRICTLY in JSON format with the following structure:\n"
            "{\n"
            "  \"title\": \"Proposed Report Title\",\n"
            "  \"sections\": [\n"
            "    {\n"
            "      \"id\": \"1\",\n"
            "      \"title\": \"Introduction\",\n"
            "      \"summary\": \"Context, objectives, and scope.\",\n"
            "      \"word_count\": \"~200 words\"\n"
            "    },\n"
            "    ...\n"
            "    {\n"
            "      \"id\": \"N\",\n"
            "      \"title\": \"References\",\n"
            "      \"summary\": \"List of sources.\",\n"
            "      \"word_count\": \"~100 words\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        user_prompt = (
            f"Task description:\n{task_description}\n\n"
            "Research notes (may be truncated):\n"
            + "\n\n".join(research_notes)
        )

        # Reuse main-agent style system prompt for tools/context if desired
        # but here we keep it simple and avoid tools.
        message_history = [
            {"role": "user", "content": user_prompt},
        ]

        response, message_history = await self.llm_client.create_message(
            system_prompt=system_prompt,
            message_history=message_history,
            tool_definitions=None,
            keep_tool_result=-1,
            step_id=0,
            task_log=self.task_log,
            agent_type="main",
        )

        outline_text = ""
        if response is not None:
            outline_text, _, _ = self.llm_client.process_llm_response(
                response, message_history, agent_type="main"
            )

        # Try to parse JSON; fall back to a minimal outline structure
        outline: Dict[str, Any]
        try:
            outline = json.loads(outline_text)
            if not isinstance(outline, dict):
                raise ValueError("Outline root is not a JSON object")
        except Exception as e:
            self.task_log.log_step(
                "warning",
                "Report | Outline",
                f"Failed to parse outline JSON, using fallback outline: {e}",
            )
            outline = {
                "title": task_description[:80],
                "sections": [
                    {
                        "id": "1",
                        "title": "Introduction",
                        "summary": "Introduction to the topic.",
                        "word_count": "~200 words",
                    },
                    {
                        "id": "2",
                        "title": "Analysis",
                        "summary": "Detailed analysis based on research.",
                        "word_count": "~800 words",
                    },
                    {
                        "id": "3",
                        "title": "Conclusion",
                        "summary": "Summary of findings.",
                        "word_count": "~300 words",
                    },
                    {
                        "id": "4",
                        "title": "References",
                        "summary": "List of sources.",
                        "word_count": "~100 words",
                    },
                ],
            }

        # Log outline JSON (truncated if very long)
        try:
            outline_str = json.dumps(outline, ensure_ascii=False, indent=2)
        except Exception:
            outline_str = str(outline)
        if len(outline_str) > 2000:
            outline_str = outline_str[:2000] + "... [truncated]"
        self.task_log.log_step(
            "info",
            "Report | Outline",
            f"Finished outline phase. Outline:\n{outline_str}",
        )
        return outline

    async def _write_sections_phase(
        self,
        task_description: str,
        outline: Dict[str, Any],
        research_notes: List[str],
    ) -> List[Dict[str, str]]:
        """Stage 3: generate each section content and return as a list of dicts."""

        self.task_log.log_step(
            "info", "Report | Sections", "Starting section writing phase",
        )

        sections = outline.get("sections") or []
        draft_sections: List[Dict[str, str]] = []

        for section in sections:
            sec_title = section.get("title", "Section")
            sec_summary = section.get("summary", "")
            sec_word_count = section.get("word_count", "")

            system_prompt = (
                "You are writing ONE section of a technical report. "
                "Write only the body for the current section in Markdown. "
                "Do not include the report title. "
                "Do not repeat the section header (e.g. '## Introduction'). "
                "Focus on content generation.\n"
                "Requirements:\n"
                "1. **Language**: Write in the SAME language as the task description and section title.\n"
                "2. **Length**: Strictly adhere to the target word count specified in the user prompt. Do not exceed it significantly. Be concise if the limit is low."
            )

            user_prompt = (
                f"Task description:\n{task_description}\n\n"
                f"Current section title: {sec_title}\n"
                f"Section summary: {sec_summary}\n"
                f"Target word count: {sec_word_count}\n\n"
                "Research notes (may be truncated):\n"
                + "\n\n".join(research_notes)
            )

            message_history = [
                {"role": "user", "content": user_prompt},
            ]

            response, message_history = await self.llm_client.create_message(
                system_prompt=system_prompt,
                message_history=message_history,
                tool_definitions=None,
                keep_tool_result=-1,
                step_id=0,
                task_log=self.task_log,
                agent_type="main",
            )

            section_text = ""
            if response is not None:
                section_text, _, _ = self.llm_client.process_llm_response(
                    response, message_history, agent_type="main"
                )

            if section_text:
                draft_sections.append({"title": sec_title, "content": section_text.strip()})

        # Log draft report preview
        preview_parts = [f"## {s['title']}\n\n{s['content']}" for s in draft_sections]
        preview = "\n\n".join(preview_parts).strip()
        if len(preview) > 2000:
            preview = preview[:2000] + "... [truncated]"
        self.task_log.log_step(
            "info",
            "Report | Sections",
            f"Finished section writing phase. Draft preview:\n{preview}",
        )
        return draft_sections

    async def _polish_phase(
        self,
        task_description: str,
        draft_sections: List[Dict[str, str]],
        research_notes: List[str],
        outline: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Stage 4: global polish + formatting to final markdown report.

        Optimized to handle long contexts by:
        1. Removing raw research notes (relying on draft content).
        2. Segmenting processing (Front matter, Sections).
        """

        self.task_log.log_step(
            "info", "Report | Polish", "Starting polish phase (segmented)",
        )

        polished_parts = []

        # 1. Generate Title and Abstract (Front Matter)
        if outline:
            system_prompt = (
                "You are a senior technical writer. "
                "Based on the task and outline, write the Title and Abstract for the report.\n"
                "Requirements:\n"
                "1. **Language**: Detect the language of the task. If Chinese, use Chinese for the Title and Abstract content. If English, use English.\n"
                "2. **Headers**: If Chinese, use '## 摘要' instead of '## Abstract'.\n"
                "Format:\n"
                "# [Report Title]\n\n"
                "## Abstract (or 摘要)\n"
                "[Abstract Content]\n\n"
                "Output ONLY the Markdown content. Do not include Introduction."
            )
            outline_str = json.dumps(outline, ensure_ascii=False, indent=2)
            user_prompt = (
                f"Task: {task_description}\n\n"
                f"Outline:\n{outline_str}\n"
            )

            message_history = [{"role": "user", "content": user_prompt}]
            response, message_history = await self.llm_client.create_message(
                system_prompt=system_prompt,
                message_history=message_history,
                tool_definitions=None,
                keep_tool_result=-1,
                step_id=0,
                task_log=self.task_log,
                agent_type="main",
            )
            if response:
                text, _, _ = self.llm_client.process_llm_response(
                    response, message_history, agent_type="main"
                )
                if text:
                    polished_parts.append(text)

        # 2. Polish Sections
        for section in draft_sections:
            title = section["title"]
            content = section["content"]
            full_text = f"## {title}\n\n{content}"

            system_prompt = (
                "You are a senior technical writer. "
                "Polish the following report section. Improve clarity, flow, and tone. "
                "Fix any grammar issues. Keep the Markdown format (headers, lists, etc.). "
                "Do NOT remove the section header.\n"
                "Requirements:\n"
                "1. **Language**: Ensure the content is in the same language as the input (Chinese or English).\n"
                "2. **Length**: Respect any implied length constraints from the content. Do not arbitrarily shorten detailed technical content unless it is repetitive.\n"
                "IMPORTANT: Output ONLY the polished Markdown content. Do not add conversational filler.\n"
                "CRITICAL: Convert all plain text URLs into Markdown hyperlinks (e.g., [Title](url))."
            )
            user_prompt = f"Section Content:\n{full_text}"

            message_history = [{"role": "user", "content": user_prompt}]
            response, message_history = await self.llm_client.create_message(
                system_prompt=system_prompt,
                message_history=message_history,
                tool_definitions=None,
                keep_tool_result=-1,
                step_id=0,
                task_log=self.task_log,
                agent_type="main",
            )
            if response:
                text, _, _ = self.llm_client.process_llm_response(
                    response, message_history, agent_type="main"
                )
                if text:
                    polished_parts.append(text)
                else:
                    polished_parts.append(full_text)
            else:
                polished_parts.append(full_text)

        final_report = "\n\n".join(polished_parts)

        # Log final polished report preview
        preview = final_report.strip()
        if len(preview) > 2000:
            preview = preview[:2000] + "... [truncated]"
        self.task_log.log_step(
            "info",
            "Report | Polish",
            f"Finished polish phase. Final report preview:\n{preview}",
        )
        return final_report

    async def _stream_update(self, event_type: str, data: dict):
        """Send streaming update in new SSE protocol format"""
        if self.stream_queue:
            try:
                stream_message = {
                    "event": event_type,
                    "data": data,
                }
                await self.stream_queue.put(stream_message)
            except Exception as e:
                logger.warning(f"Failed to send stream update: {e}")

    async def _stream_start_workflow(self, user_input: str) -> str:
        """Send start_of_workflow event"""
        workflow_id = str(uuid.uuid4())
        await self._stream_update(
            "start_of_workflow",
            {
                "workflow_id": workflow_id,
                "input": [
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ],
            },
        )
        return workflow_id

    async def _stream_end_workflow(self, workflow_id: str):
        """Send end_of_workflow event"""
        await self._stream_update(
            "end_of_workflow",
            {
                "workflow_id": workflow_id,
            },
        )

    async def _stream_show_error(self, error: str):
        """Send show_error event"""
        await self._stream_tool_call("show_error", {"error": error})
        if self.stream_queue:
            try:
                await self.stream_queue.put(None)
            except Exception as e:
                logger.warning(f"Failed to send show_error: {e}")

    async def _stream_start_agent(self, agent_name: str, display_name: str = None):
        """Send start_of_agent event"""
        agent_id = str(uuid.uuid4())
        await self._stream_update(
            "start_of_agent",
            {
                "agent_name": agent_name,
                "display_name": display_name,
                "agent_id": agent_id,
            },
        )
        return agent_id

    async def _stream_end_agent(self, agent_name: str, agent_id: str):
        """Send end_of_agent event"""
        await self._stream_update(
            "end_of_agent",
            {
                "agent_name": agent_name,
                "agent_id": agent_id,
            },
        )

    async def _stream_start_llm(self, agent_name: str, display_name: str = None):
        """Send start_of_llm event"""
        await self._stream_update(
            "start_of_llm",
            {
                "agent_name": agent_name,
                "display_name": display_name,
            },
        )

    async def _stream_end_llm(self, agent_name: str):
        """Send end_of_llm event"""
        await self._stream_update(
            "end_of_llm",
            {
                "agent_name": agent_name,
            },
        )

    async def _stream_message(self, message_id: str, delta_content: str):
        """Send message event"""
        await self._stream_update(
            "message",
            {
                "message_id": message_id,
                "delta": {
                    "content": delta_content,
                },
            },
        )

    async def _stream_tool_call(
        self,
        tool_name: str,
        payload: dict,
        streaming: bool = False,
        tool_call_id: str = None,
    ) -> str:
        """Send tool_call event"""
        if not tool_call_id:
            tool_call_id = str(uuid.uuid4())

        if streaming:
            for key, value in payload.items():
                await self._stream_update(
                    "tool_call",
                    {
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "delta_input": {key: value},
                    },
                )
        else:
            # Send complete tool call
            await self._stream_update(
                "tool_call",
                {
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "tool_input": payload,
                },
            )

        return tool_call_id

    def get_scrape_result(self, result: str) -> str:
        """
        Process scrape result and truncate if too long to support more conversation turns.
        """
        SCRAPE_MAX_LENGTH = 20000
        try:
            scrape_result_dict = json.loads(result)
            text = scrape_result_dict.get("text")
            if text and len(text) > SCRAPE_MAX_LENGTH:
                text = text[:SCRAPE_MAX_LENGTH]
            return json.dumps({"text": text}, ensure_ascii=False)
        except json.JSONDecodeError:
            if isinstance(result, str) and len(result) > SCRAPE_MAX_LENGTH:
                result = result[:SCRAPE_MAX_LENGTH]
            return result

    def post_process_tool_call_result(self, tool_name, tool_call_result: dict):
        """Process tool call results"""
        # Only in demo mode: truncate scrape results to 20,000 chars
        # to support more conversation turns. Skipped in perf tests to avoid loss.
        if os.environ.get("DEMO_MODE") == "1":
            if "result" in tool_call_result and tool_name in [
                "scrape",
                "scrape_website",
            ]:
                tool_call_result["result"] = self.get_scrape_result(
                    tool_call_result["result"]
                )
        return tool_call_result

    def _get_query_str_from_tool_call(
        self, tool_name: str, arguments: dict
    ) -> Optional[str]:
        """
        Extracts the query string from tool call arguments based on tool_name.
        Supports search_and_browse, google_search, sougou_search, and scrape_website.
        """
        if tool_name == "search_and_browse":
            return arguments.get("subtask")
        elif tool_name == "google_search":
            return arguments.get("q")
        elif tool_name == "sougou_search":
            return arguments.get("Query")
        elif tool_name == "scrape_website":
            return arguments.get("url")
        return None

    async def _handle_llm_call(
        self,
        system_prompt,
        message_history,
        tool_definitions,
        step_id: int,
        purpose: str = "",
        keep_tool_result: int = -1,
        agent_type: str = "main",
    ) -> Tuple[Optional[str], bool, Optional[Any], List[Dict[str, Any]]]:
        """Unified LLM call and logging processing
        Returns:
            Tuple[Optional[str], bool, Optional[Any], List[Dict[str, Any]]]:
                (response_text, should_break, tool_calls_info, message_history)
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
            if ErrorBox.is_error_box(response):
                await self._stream_show_error(str(response))
                response = None
            should_break = False
            if ResponseBox.is_response_box(response):
                if response.has_extra_info():
                    extra_info = response.get_extra_info()
                    if extra_info.get("should_break", False):
                        should_break = True
                    if extra_info.get("warning_msg"):
                        await self._stream_show_error(
                            extra_info.get("warning_msg", "Empty warning message")
                        )

                response = response.get_response()
            # Check if response is None (indicating an error occurred)
            if response is None:
                self.task_log.log_step(
                    "error",
                    f"{purpose} | LLM Call Failed",
                    f"{purpose} failed - no response received",
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
                "info",
                f"{purpose} | LLM Call",
                "completed successfully",
            )
            return (
                assistant_response_text,
                should_break,
                tool_calls_info,
                message_history,
            )

        except Exception as e:
            self.task_log.log_step(
                "error",
                f"{purpose} | LLM Call ERROR",
                f"{purpose} error: {str(e)}",
            )
            # Return empty response with should_break=True to indicate error
            return "", True, None, message_history

    async def run_sub_agent(
        self, sub_agent_name, task_description, keep_tool_result: int = -1
    ):
        """Run sub agent"""
        self.task_log.log_step(
            "info", f"{sub_agent_name} | Start Task", f"Starting {sub_agent_name}"
        )
        use_cn_prompt = os.getenv("USE_CN_PROMPT", "0")
        if use_cn_prompt == "1":
            task_description += "\n\n请给出该子任务的答案，并提供详细的依据或支持信息。"
        else:
            task_description += "\n\nPlease provide the answer and detailed supporting information of the subtask given to you."
        self.task_log.log_step(
            "info",
            f"{sub_agent_name} | Task Description",
            f"Subtask: {task_description}",
        )

        # Stream sub-agent start
        display_name = sub_agent_name.replace("agent-", "")
        sub_agent_id = await self._stream_start_agent(display_name)
        await self._stream_start_llm(display_name)

        # Start new sub-agent session
        self.task_log.start_sub_agent_session(sub_agent_name, task_description)

        # Simplified initial user content (no file attachments)
        initial_user_content = task_description
        message_history = [{"role": "user", "content": initial_user_content}]

        # Get sub-agent tool definitions
        if not self.sub_agent_tool_definitions:
            tool_definitions = await self._list_sub_agent_tools()
            tool_definitions = tool_definitions.get(sub_agent_name, {})
        else:
            tool_definitions = self.sub_agent_tool_definitions[sub_agent_name]
        self.task_log.log_step(
            "info",
            f"{sub_agent_name} | Get Tool Definitions",
            f"Number of tools: {len(tool_definitions)}",
        )

        if not tool_definitions:
            self.task_log.log_step(
                "warning",
                f"{sub_agent_name} | No Tools",
                "No tool definitions available.",
            )

        # Generate sub-agent system prompt
        system_prompt = self.llm_client.generate_agent_system_prompt(
            date=date.today(),
            mcp_servers=tool_definitions,
        ) + generate_agent_specific_system_prompt(agent_type=sub_agent_name)

        # Limit sub-agent turns
        max_turns = self.cfg.agent.sub_agents[sub_agent_name].max_turns
        turn_count = 0
        should_hard_stop = False

        while turn_count < max_turns:
            turn_count += 1
            self.task_log.log_step(
                "info",
                f"{sub_agent_name} | Turn: {turn_count}",
                f"Starting turn {turn_count}.",
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
                f"{sub_agent_name} | Turn: {turn_count}",
                agent_type=sub_agent_name,
            )

            if should_break:
                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | LLM Call",
                    "should break is True, breaking the loop",
                )
                break

            # Process LLM response
            elif assistant_response_text:
                text_response = extract_llm_response_text(assistant_response_text)
                if text_response:
                    await self._stream_tool_call("show_text", {"text": text_response})
                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | LLM Call",
                    "completed successfully",
                )

            else:
                # LLM call failed, end current turn
                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | LLM Call",
                    "LLM call failed",
                )
                break

            # Use tool calls parsed from LLM response
            if not tool_calls:
                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | LLM Call",
                    f"No tool calls found in {sub_agent_name}, ending on turn {turn_count}",
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
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | Tool Call",
                    f"Executing {tool_name} on {server_name}",
                )

                call_start_time = time.time()
                try:
                    tool_call_id = await self._stream_tool_call(tool_name, arguments)
                    query_str = self._get_query_str_from_tool_call(tool_name, arguments)
                    if query_str:
                        cache_name = sub_agent_id + "_" + tool_name
                        self.used_queries.setdefault(cache_name, defaultdict(int))
                        count = self.used_queries[cache_name][query_str]
                        if count > 0:
                            tool_result = {
                                "server_name": server_name,
                                "tool_name": tool_name,
                                "result": f"The query '{query_str}' has already been used in previous {tool_name}. Please try a different query or keyword.",
                            }
                            if count >= self.max_repeat_queries:
                                should_hard_stop = True
                        else:
                            tool_result = await self.sub_agent_tool_managers[
                                sub_agent_name
                            ].execute_tool_call(server_name, tool_name, arguments)
                        if "error" not in tool_result:
                            self.used_queries[cache_name][query_str] += 1
                    else:
                        tool_result = await self.sub_agent_tool_managers[
                            sub_agent_name
                        ].execute_tool_call(server_name, tool_name, arguments)

                    # Only in demo mode: truncate scrape results to 20,000 chars
                    tool_result = self.post_process_tool_call_result(
                        tool_name, tool_result
                    )
                    result = (
                        tool_result.get("result")
                        if tool_result.get("result")
                        else tool_result.get("error")
                    )
                    await self._stream_tool_call(
                        tool_name, {"result": result}, tool_call_id=tool_call_id
                    )
                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    self.task_log.log_step(
                        "info",
                        f"{sub_agent_name} | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} completed in {call_duration_ms}ms",
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
                        "error",
                        f"{sub_agent_name} | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} failed to execute: {str(e)}",
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
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | Context Limit Reached",
                    "Context limit reached, triggering summary",
                )
                break

            # If a repeat occurs, terminate early to speed up task completion
            if should_hard_stop:
                break

        # Continue processing
        self.task_log.log_step(
            "info",
            f"{sub_agent_name} | Main Loop Completed",
            f"Main loop completed after {turn_count} turns",
        )

        # Record browsing agent loop end
        if turn_count >= max_turns:
            self.task_log.log_step(
                "info",
                f"{sub_agent_name} | Max Turns Reached / Context Limit Reached",
                f"Reached maximum turns ({max_turns}) or context limit reached",
            )

        else:
            self.task_log.log_step(
                "info",
                f"{sub_agent_name} | Main Loop Completed",
                f"Main loop completed after {turn_count} turns",
            )

        # Final summary
        self.task_log.log_step(
            "info",
            f"{sub_agent_name} | Final Summary",
            f"Generating {sub_agent_name} final summary",
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

        await self._stream_tool_call(
            "Partial Summary", {}, tool_call_id=str(uuid.uuid4())
        )

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
            f"{sub_agent_name} | Final summary",
            keep_tool_result=keep_tool_result,
            agent_type=sub_agent_name,
        )

        if final_answer_text:
            self.task_log.log_step(
                "info",
                f"{sub_agent_name} | Final Answer",
                "Final answer generated successfully",
            )

        else:
            final_answer_text = (
                f"No final answer generated by sub agent {sub_agent_name}."
            )
            self.task_log.log_step(
                "error",
                f"{sub_agent_name} | Final Answer",
                "Unable to generate final answer",
            )

        self.task_log.sub_agent_message_history_sessions[
            self.task_log.current_sub_agent_session_id
        ] = {"system_prompt": system_prompt, "message_history": message_history}

        self.task_log.save()

        self.task_log.end_sub_agent_session(sub_agent_name)

        # Remove thinking content in tool response
        # For the return result of a sub-agent, the content within the `<think>` tags is unnecessary in any case.
        final_answer_text = final_answer_text.split("<think>")[-1].strip()
        final_answer_text = final_answer_text.split("</think>")[-1].strip()

        # Stream sub-agent end
        await self._stream_end_llm(display_name)
        await self._stream_end_agent(display_name, sub_agent_id)

        # Return final answer instead of conversation log, so main agent can use it directly
        return final_answer_text

    async def run_main_agent(
        self, task_description, task_file_paths=None, task_id="default_task"
    ):
        """Execute the main end-to-end task"""
        workflow_id = await self._stream_start_workflow(task_description)
        keep_tool_result = int(self.cfg.agent.keep_tool_result)

        self.task_log.log_step("info", "Main Agent", f"Start task with id: {task_id}")
        self.task_log.log_step(
            "info", "Main Agent", f"Task description: {task_description}"
        )
        if task_file_paths:
            self.task_log.log_step(
                "info", "Main Agent", f"Associated files: {task_file_paths}"
            )

        # 1. Process input
        initial_user_content, processed_task_desc = process_input(
            task_description, task_file_paths
        )
        message_history = [{"role": "user", "content": initial_user_content}]

        # Record initial user input
        user_input = processed_task_desc
        if task_file_paths:
            user_input += f"\n[Attached files: {task_file_paths}]"

        # 2. Get tool definitions
        if not self.tool_definitions:
            tool_definitions = (
                await self.main_agent_tool_manager.get_all_tool_definitions()
            )
            tool_definitions += expose_sub_agents_as_tools(self.cfg.agent.sub_agents)
        else:
            tool_definitions = self.tool_definitions
        if not tool_definitions:
            self.task_log.log_step(
                "warning",
                "Main Agent | Tool Definitions",
                "Warning: No tool definitions found. LLM cannot use any tools.",
            )

        self.task_log.log_step(
            "info", "Main Agent", f"Number of tools: {len(tool_definitions)}"
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
        should_hard_stop = False

        self.current_agent_id = await self._stream_start_agent("main")
        await self._stream_start_llm("main")
        while turn_count < max_turns:
            turn_count += 1
            self.task_log.log_step(
                "info",
                f"Main Agent | Turn: {turn_count}",
                f"Starting turn {turn_count}",
            )
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
                text_response = extract_llm_response_text(assistant_response_text)
                if text_response:
                    await self._stream_tool_call("show_text", {"text": text_response})
                if should_break:
                    self.task_log.log_step(
                        "info",
                        f"Main Agent | Turn: {turn_count} | LLM Call",
                        "should break is True, breaking the loop",
                    )
                    break

            else:
                self.task_log.log_step(
                    "info",
                    f"Main Agent | Turn: {turn_count} | LLM Call",
                    "No valid response from LLM, breaking the loop",
                )
                break

            if not tool_calls:
                self.task_log.log_step(
                    "info",
                    f"Main Agent | Turn: {turn_count} | LLM Call",
                    "LLM did not request tool usage, ending process.",
                )
                break

            # 7. Execute tool calls (execute in order)
            tool_calls_data = []
            all_tool_results_content_with_id = []

            self.task_log.log_step(
                "info",
                f"Main Agent | Turn: {turn_count} | Tool Calls",
                f"Number of tool calls detected: {len(tool_calls)}",
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
                        await self._stream_end_llm("main")
                        await self._stream_end_agent("main", self.current_agent_id)
                        query_str = self._get_query_str_from_tool_call(
                            tool_name, arguments
                        )
                        if query_str:
                            cache_name = self.current_agent_id + "_" + tool_name
                            self.used_queries.setdefault(cache_name, defaultdict(int))
                            count = self.used_queries[cache_name][query_str]
                            if count > 0:
                                sub_agent_result = f"The query '{query_str}' has already been used in previous {tool_name}. Please try a different query or keyword."
                                if count >= self.max_repeat_queries:
                                    should_hard_stop = True
                            else:
                                sub_agent_result = await self.run_sub_agent(
                                    server_name, arguments["subtask"], keep_tool_result
                                )
                            self.used_queries[cache_name][query_str] += 1
                        else:
                            sub_agent_result = await self.run_sub_agent(
                                server_name, arguments["subtask"], keep_tool_result
                            )
                        tool_result = {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "result": sub_agent_result,
                        }
                        self.current_agent_id = await self._stream_start_agent(
                            "main", display_name="Summarizing"
                        )
                        await self._stream_start_llm("main", display_name="Summarizing")
                    else:
                        tool_call_id = await self._stream_tool_call(
                            tool_name, arguments
                        )
                        query_str = self._get_query_str_from_tool_call(
                            tool_name, arguments
                        )
                        if query_str:
                            cache_name = self.current_agent_id + "_" + tool_name
                            self.used_queries.setdefault(cache_name, defaultdict(int))
                            count = self.used_queries[cache_name][query_str]
                            if count > 0:
                                tool_result = {
                                    "server_name": server_name,
                                    "tool_name": tool_name,
                                    "result": f"The query '{query_str}' has already been used in previous {tool_name}. Please try a different query or keyword.",
                                }
                                if count >= self.max_repeat_queries:
                                    should_hard_stop = True
                            else:
                                tool_result = await self.main_agent_tool_manager.execute_tool_call(
                                    server_name=server_name,
                                    tool_name=tool_name,
                                    arguments=arguments,
                                )
                            if "error" not in tool_result:
                                self.used_queries[cache_name][query_str] += 1
                        else:
                            tool_result = (
                                await self.main_agent_tool_manager.execute_tool_call(
                                    server_name=server_name,
                                    tool_name=tool_name,
                                    arguments=arguments,
                                )
                            )
                        # Only in demo mode: truncate scrape results to 20,000 chars
                        tool_result = self.post_process_tool_call_result(
                            tool_name, tool_result
                        )
                        result = (
                            tool_result.get("result")
                            if tool_result.get("result")
                            else tool_result.get("error")
                        )
                        await self._stream_tool_call(
                            tool_name, {"result": result}, tool_call_id=tool_call_id
                        )

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
                        "info",
                        f"Main Agent | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} completed in {call_duration_ms}ms",
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
                        "error",
                        f"Main Agent | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} failed to execute: {str(e)}",
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
                    "warning",
                    f"Main Agent | Turn: {turn_count} | Context Limit Reached",
                    "Context limit reached, triggering summary",
                )
                break

            if should_hard_stop:
                break

        await self._stream_end_llm("main")
        await self._stream_end_agent("main", self.current_agent_id)

        # Record main loop end
        if turn_count >= max_turns:
            self.task_log.log_step(
                "warning",
                "Main Agent | Max Turns Reached / Context Limit Reached",
                f"Reached maximum turns ({max_turns}) or context limit reached",
            )

        else:
            self.task_log.log_step(
                "info",
                "Main Agent | Main Loop Completed",
                f"Main loop completed after {turn_count} turns",
            )

        # Final summary
        self.task_log.log_step(
            "info", "Main Agent | Final Summary", "Generating final summary"
        )

        self.current_agent_id = await self._stream_start_agent("Final Summary")
        await self._stream_start_llm("Final Summary")

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
                "info",
                "Main Agent | Final Answer",
                "Final answer generated successfully",
            )

            # Log the final answer
            self.task_log.log_step(
                "info",
                "Main Agent | Final Answer",
                f"Final answer content:\n\n{final_answer_text}",
            )

        else:
            final_answer_text = "No final answer generated."
            self.task_log.log_step(
                "error",
                "Main Agent | Final Answer",
                "Unable to generate final answer",
            )

        final_summary, final_boxed_answer, usage_log = (
            self.output_formatter.format_final_summary_and_log(
                final_answer_text, self.llm_client
            )
        )

        await self._stream_tool_call("show_text", {"text": final_boxed_answer})
        await self._stream_end_llm("Final Summary")
        await self._stream_end_agent("Final Summary", self.current_agent_id)
        await self._stream_end_workflow(workflow_id)

        self.task_log.log_step(
            "info", "Main Agent | Usage Calculation", f"Usage log: {usage_log}"
        )

        self.task_log.log_step(
            "info",
            "Main Agent | Final boxed answer",
            f"Final boxed answer:\n\n{final_boxed_answer}",
        )

        self.task_log.log_step(
            "info",
            "Main Agent | Task Completed",
            f"Main agent task {task_id} completed successfully",
        )

        return final_summary, final_boxed_answer
