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

import asyncio
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

# Import from the new modular structure
from src.core.pipeline import (
    create_pipeline_components,
    execute_task_pipeline,
)

logger = logging.getLogger()

async def amain(cfg: DictConfig) -> None:
    """Asynchronous main function."""
    logger.info(OmegaConf.to_yaml(cfg))
    # Create pipeline components using the factory function
    main_agent_tool_manager, sub_agent_tool_managers, output_formatter = (
        create_pipeline_components(cfg)
    )

    # Define task parameters
    task_file_name = ""
    task_id = "task_example"
    task_original_name = "task_example"
    task_description = "What percentage of the total penguin population according to the upper estimates on english Wikipedia at the end of 2012 is made up by the penguins in this file that don't live on Dream Island and have beaks longer than 42mm? Round to the nearest five decimal places."
    task_description = "What is the title of today's arxiv paper in computer science?"
    task_file_name = ""

    # Execute task using the pipeline
    final_summary, final_boxed_answer, log_file_path = await execute_task_pipeline(
        cfg=cfg,
        task_original_name=task_original_name,
        task_id=task_id,
        task_file_name=task_file_name,
        task_description=task_description,
        main_agent_tool_manager=main_agent_tool_manager,
        sub_agent_tool_managers=sub_agent_tool_managers,
        output_formatter=output_formatter,
        log_dir=cfg.debug_dir,
    )


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))


if __name__ == "__main__":
    main()
