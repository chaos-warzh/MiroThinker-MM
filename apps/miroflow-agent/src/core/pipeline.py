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


import traceback
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from miroflow_tools.manager import ToolManager
from miroflow_tracing import trace
from omegaconf import DictConfig

from ..config.settings import (
    create_mcp_server_parameters,
    get_env_info,
)
from .orchestrator import Orchestrator
from ..io.output_formatter import OutputFormatter
from ..llm.client import LLMClient
from ..logging.task_logger import (
    TaskLog,
    get_utc_plus_8_time_z_format,
    logger,
)
from ..logging.tracing_processor import setup_file_trace_processor


async def execute_task_pipeline(
    cfg: DictConfig,
    task_original_name: str,
    task_id: str,
    task_description: str,
    task_file_name: str,
    main_agent_tool_manager: ToolManager,
    sub_agent_tool_managers: List[Dict[str, ToolManager]],
    output_formatter: OutputFormatter,
    ground_truth: Optional[Any] = None,
    log_dir: str = "logs",
):
    """
    Executes the full pipeline for a single task.

    Args:
        cfg: The Hydra configuration object.
        task_description: The description of the task for the LLM.
        task_file_name: The path to an associated file (optional).
        task_id: A unique identifier for this task run (used for logging).
        main_agent_tool_manager: An initialized main agent ToolManager instance.
        sub_agent_tool_managers: A dictionary of initialized sub-agent ToolManager instances.
        output_formatter: An initialized OutputFormatter instance.
        ground_truth: The ground truth for the task (optional).
        log_dir: The directory to save the task log (default: "logs").

    Returns:
        A tuple containing:
        - A string with the final execution log and summary, or an error message.
        - The final boxed answer.
        - The path to the log file.
    """
    logger.info(f"\n--- Starting Task Execution: {task_id} ---")

    # Create task log
    task_log = TaskLog(
        log_dir=log_dir,
        task_original_name=task_original_name,
        task_id=task_id,
        start_time=get_utc_plus_8_time_z_format(),
        input={"task_description": task_description, "task_file_name": task_file_name},
        env_info=get_env_info(cfg),
        ground_truth=ground_truth,
    )

    traces = []
    try:
        # Initialize LLM client
        llm_client = LLMClient(task_id=task_id, cfg=cfg)

        # Initialize orchestrator
        orchestrator = Orchestrator(
            main_agent_tool_manager=main_agent_tool_manager,
            sub_agent_tool_managers=sub_agent_tool_managers,
            llm_client=llm_client,
            output_formatter=output_formatter,
            cfg=cfg,
            task_log=task_log,
        )

        trace_processor = setup_file_trace_processor()
        with trace(workflow_name="benchmark_workflow", trace_id=task_id):
            final_summary, final_boxed_answer = await orchestrator.run_main_agent(
                task_description=task_description,
                task_file_name=task_file_name,
                task_id=task_id,
            )

        traces = trace_processor.get_and_clear_traces(task_id=task_id)
        if traces:
            task_log.trace_data = _process_spans_for_summary(traces[0], cfg)

        llm_client.close()

        task_log.final_boxed_answer = final_boxed_answer
        task_log.status = "success"

        log_file_path = task_log.save()
        return final_summary, final_boxed_answer, log_file_path

    except Exception as e:
        error_details = traceback.format_exc()
        logger.warning(
            f"\n{'!' * 20} An error occurred during task {task_id}: {'!' * 20}"
        )
        logger.error(error_details, exc_info=True)

        error_message = (
            f"Error executing task {task_id}:\n"
            f"Description: {task_description}\n"
            f"File: {task_file_name}\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Details:\n{error_details}"
        )

        task_log.status = "failed"
        task_log.error = error_details

        log_file_path = task_log.save()

        return error_message, "", log_file_path

    finally:
        task_log.end_time = get_utc_plus_8_time_z_format()

        # Record task summary to structured log
        task_log.log_step(
            "task_execution_finished",
            f"Task {task_id} execution completed with status: {task_log.status}",
        )
        task_log.log_step(
            "console_summary_display", "Displaying task summary to console"
        )
        task_log.save()

        logger.info(f"--- Finished Task Execution: {task_id} ---")


def _process_spans_for_summary(trace_obj, cfg):
    """
    Processes a trace object to generate a structured performance summary.
    """
    # --- 1. Pre-computation and tree building ---
    spans_by_id = {span.span_id: span for span in getattr(trace_obj, "spans", [])}
    children_by_parent_id = defaultdict(list)
    for span in spans_by_id.values():
        children_by_parent_id[span.parent_id].append(span)

    def get_sub_agent_spans(
        sub_agent_name: str,
        spans_by_id: Dict[str, Any],
        children_by_parent_id: Dict[str, List[Any]],
    ) -> Set[str]:
        # --- 2. Find all sub-agent spans ---
        sub_agent_span_ids = set()
        queue = []
        # Find the root sub-agent spans and use them as the entry point for the search
        for span in spans_by_id.values():
            span_name = getattr(span.span_data, "name", "")
            if sub_agent_name in span_name:
                # The children of this span belong to the sub-agent's context
                queue.extend(children_by_parent_id.get(span.span_id, []))

        # Start the BFS to find all descendant spans
        head = 0
        while head < len(queue):
            current_span = queue[head]
            head += 1
            if current_span.span_id not in sub_agent_span_ids:
                sub_agent_span_ids.add(current_span.span_id)
                queue.extend(children_by_parent_id.get(current_span.span_id, []))

        return sub_agent_span_ids

    sub_agent_span_ids_dict = {
        sub_agent_name: get_sub_agent_spans(
            sub_agent_name, spans_by_id, children_by_parent_id
        )
        for sub_agent_name in cfg.agent.sub_agents.keys()
    }

    # --- 3. Pre-calculate all span durations ---
    for span in spans_by_id.values():
        duration = (
            datetime.fromisoformat(span.ended_at)
            - datetime.fromisoformat(span.started_at)
        ).total_seconds()
        span.duration = duration  # Attach for later use

    # --- 4. Final Wall Time Calculations ---
    wall_time_total = 0
    if hasattr(trace_obj, "started_at") and hasattr(trace_obj, "ended_at"):
        wall_time_total = (
            datetime.fromisoformat(trace_obj.ended_at)
            - datetime.fromisoformat(trace_obj.started_at)
        ).total_seconds()

    # Find top-level spans for main agent and all sub-agent root spans
    main_agent_top_level_spans = [
        s
        for s in spans_by_id.values()
        if s.parent_id is None
        and not any(
            sub_agent_name in getattr(s.span_data, "name", "")
            for sub_agent_name in cfg.agent.sub_agents.keys()
        )
    ]
    sub_agent_root_spans_dict = {
        sub_agent_name: [
            s
            for s in spans_by_id.values()
            if s.parent_id is None
            and sub_agent_name in getattr(s.span_data, "name", "")
        ]
        for sub_agent_name in cfg.agent.sub_agents.keys()
    }

    # Calculate main agent wall time
    wall_time_main_agent_llm = sum(
        s.duration
        for s in main_agent_top_level_spans
        if getattr(s.span_data, "name", "generation_span") == "generation_span"
    )
    wall_time_main_agent_tool = sum(
        s.duration
        for s in main_agent_top_level_spans
        if getattr(s.span_data, "name", "generation_span") != "generation_span"
    )
    wall_time_main_agent = wall_time_main_agent_llm + wall_time_main_agent_tool

    # Calculate sub-agent wall time and its internal breakdown
    def calculate_sub_agent_wall_time(
        sub_agent_name: str,
    ) -> tuple[float, float, float]:
        wall_time_sub_agent = sum(
            s.duration for s in sub_agent_root_spans_dict[sub_agent_name]
        )
        wall_time_sub_agent_llm = 0
        wall_time_sub_agent_tool = 0
        for root_span in sub_agent_root_spans_dict[sub_agent_name]:
            sub_agent_children = children_by_parent_id.get(root_span.span_id, [])
            wall_time_sub_agent_llm += sum(
                s.duration
                for s in sub_agent_children
                if getattr(s.span_data, "name", "generation_span") == "generation_span"
            )
            wall_time_sub_agent_tool += sum(
                s.duration
                for s in sub_agent_children
                if getattr(s.span_data, "name", "generation_span") != "generation_span"
            )
        return wall_time_sub_agent, wall_time_sub_agent_llm, wall_time_sub_agent_tool

    wall_time_sub_agent_dict = {
        sub_agent_name: calculate_sub_agent_wall_time(sub_agent_name)
        for sub_agent_name in cfg.agent.sub_agents.keys()
    }

    # Calculate total wall times for summary
    wall_time_llm = wall_time_main_agent_llm + sum(
        wall_time_sub_agent_dict[sub_agent_name][1]
        for sub_agent_name in cfg.agent.sub_agents.keys()
    )
    wall_time_tool = wall_time_main_agent_tool + sum(
        wall_time_sub_agent_dict[sub_agent_name][2]
        for sub_agent_name in cfg.agent.sub_agents.keys()
    )

    # Orchestration breakdown
    orchestration_sub_agent_dict = {
        sub_agent_name: wall_time_sub_agent_dict[sub_agent_name][0]
        - (
            wall_time_sub_agent_dict[sub_agent_name][1]
            + wall_time_sub_agent_dict[sub_agent_name][2]
        )
        for sub_agent_name in cfg.agent.sub_agents.keys()
    }

    orchestration_main_agent = (
        wall_time_total
        - wall_time_main_agent
        - sum(
            wall_time_sub_agent_dict[sub_agent_name][0]
            for sub_agent_name in cfg.agent.sub_agents.keys()
        )
    )
    orchestration_total = orchestration_main_agent + sum(
        orchestration_sub_agent_dict.values()
    )

    # --- 5. Tool Workload Breakdown ---
    tool_workload_breakdown = defaultdict(float)
    function_spans = [
        s
        for s in spans_by_id.values()
        if getattr(s.span_data, "name", "generation_span") != "generation_span"
        and "agent-" not in getattr(s.span_data, "name", "generation_span")
    ]
    for span in function_spans:
        tool_workload_breakdown[getattr(span.span_data, "name")] += span.duration

    return {
        "trace_id": trace_obj.trace_id,
        "workflow_name": trace_obj.name,
        "performance_summary": {
            "total_wall_time": wall_time_total,
            "primary_breakdown": {
                "main-agent": {
                    "total": wall_time_main_agent + orchestration_main_agent,
                    "llm": wall_time_main_agent_llm,
                    "tool": wall_time_main_agent_tool,
                    "orchestration": orchestration_main_agent,
                },
                **{
                    sub_agent_name: {
                        "total": wall_time_sub_agent_dict[sub_agent_name][0],
                        "llm": wall_time_sub_agent_dict[sub_agent_name][1],
                        "tool": wall_time_sub_agent_dict[sub_agent_name][2],
                        "orchestration": orchestration_sub_agent_dict[sub_agent_name],
                    }
                    for sub_agent_name in cfg.agent.sub_agents.keys()
                },
            },
            "cross_cutting_breakdown": {
                "total_llm_time": wall_time_llm,
                "total_tool_time": wall_time_tool,
                "total_orchestration_time": orchestration_total,
            },
        },
        "tool_workload_breakdown": dict(tool_workload_breakdown),
        "spans": [
            {
                "name": getattr(s.span_data, "name", "generation_span"),
                "agent_context": "main-agent"
                if s.parent_id is None
                else next(
                    (
                        sub_agent_name
                        for sub_agent_name in cfg.agent.sub_agents.keys()
                        if s.span_id in sub_agent_span_ids_dict[sub_agent_name]
                    ),
                    "main-agent",
                ),
                "duration_seconds": s.duration,
                "start_time": s.started_at,
                "end_time": s.ended_at,
            }
            for s in spans_by_id.values()
        ],
    }


def create_pipeline_components(cfg: DictConfig):
    """
    Creates and initializes the core components of the agent pipeline.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        Tuple of (main_agent_tool_manager, sub_agent_tool_managers, output_formatter)
    """
    # Create ToolManagers for main agent and sub-agents
    main_agent_mcp_server_configs, main_agent_blacklist = create_mcp_server_parameters(
        cfg, cfg.agent.main_agent
    )
    main_agent_tool_manager = ToolManager(
        main_agent_mcp_server_configs,
        tool_blacklist=main_agent_blacklist,
    )

    sub_agent_tool_managers = {}
    for sub_agent in cfg.agent.sub_agents:
        sub_agent_mcp_server_configs, sub_agent_blacklist = (
            create_mcp_server_parameters(cfg, cfg.agent.sub_agents[sub_agent])
        )
        sub_agent_tool_manager = ToolManager(
            sub_agent_mcp_server_configs,
            tool_blacklist=sub_agent_blacklist,
        )
        sub_agent_tool_managers[sub_agent] = sub_agent_tool_manager

    # Create OutputFormatter
    output_formatter = OutputFormatter()

    return main_agent_tool_manager, sub_agent_tool_managers, output_formatter
