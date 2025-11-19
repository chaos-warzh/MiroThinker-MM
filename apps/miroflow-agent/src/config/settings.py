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
import sys

from dotenv import load_dotenv
from mcp import StdioServerParameters
from omegaconf import DictConfig

# Load environment variables from .env file
load_dotenv()

# API for Google Search
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
SERPER_BASE_URL = os.environ.get("SERPER_BASE_URL", "https://google.serper.dev")

# API for Web Scraping
JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_BASE_URL = os.environ.get("JINA_BASE_URL", "https://r.jina.ai")

# API for Linux Sandbox
E2B_API_KEY = os.environ.get("E2B_API_KEY")

# API for Open-Source Audio Transcription Tool
WHISPER_BASE_URL = os.environ.get("WHISPER_BASE_URL")
WHISPER_API_KEY = os.environ.get("WHISPER_API_KEY")
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME")

# API for Open-Source VQA Tool
VISION_API_KEY = os.environ.get("VISION_API_KEY")
VISION_BASE_URL = os.environ.get("VISION_BASE_URL")
VISION_MODEL_NAME = os.environ.get("VISION_MODEL_NAME")

# API for Open-Source Reasoning Tool
REASONING_API_KEY = os.environ.get("REASONING_API_KEY")
REASONING_BASE_URL = os.environ.get("REASONING_BASE_URL")
REASONING_MODEL_NAME = os.environ.get("REASONING_MODEL_NAME")

# API for Claude Sonnet 3.7 as Commercial Tools
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

# API Keys for Commercial Tools
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# API for Sougou Search
TENCENTCLOUD_SECRET_ID = os.environ.get("TENCENTCLOUD_SECRET_ID")
TENCENTCLOUD_SECRET_KEY = os.environ.get("TENCENTCLOUD_SECRET_KEY")

# ==================== Enhanced Vision Understanding (VLLM) ====================
# Multi-provider Vision Language Model support for advanced image understanding
# Similar to Google Search API configuration pattern

# Primary VLLM Provider - supports: "openai", "anthropic", "qwen", "custom"
VLLM_PROVIDER = os.environ.get("VLLM_PROVIDER", "openai")

# OpenAI-compatible VLLM configuration (GPT-4V, GPT-4o, etc.)
VLLM_OPENAI_API_KEY = os.environ.get("VLLM_OPENAI_API_KEY") or OPENAI_API_KEY
VLLM_OPENAI_BASE_URL = os.environ.get("VLLM_OPENAI_BASE_URL") or OPENAI_BASE_URL
VLLM_OPENAI_MODEL = os.environ.get("VLLM_OPENAI_MODEL", "gpt-4-vision")

# Anthropic Claude Vision configuration
VLLM_ANTHROPIC_API_KEY = os.environ.get("VLLM_ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY
VLLM_ANTHROPIC_BASE_URL = os.environ.get("VLLM_ANTHROPIC_BASE_URL") or ANTHROPIC_BASE_URL
VLLM_ANTHROPIC_MODEL = os.environ.get("VLLM_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

# Qwen Vision configuration (Open-source option)
VLLM_QWEN_API_KEY = os.environ.get("VLLM_QWEN_API_KEY") or VISION_API_KEY
VLLM_QWEN_BASE_URL = os.environ.get("VLLM_QWEN_BASE_URL") or VISION_BASE_URL
VLLM_QWEN_MODEL = os.environ.get("VLLM_QWEN_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")

# Custom VLLM configuration (for self-hosted or other providers)
VLLM_CUSTOM_API_KEY = os.environ.get("VLLM_CUSTOM_API_KEY")
VLLM_CUSTOM_BASE_URL = os.environ.get("VLLM_CUSTOM_BASE_URL")
VLLM_CUSTOM_MODEL = os.environ.get("VLLM_CUSTOM_MODEL")

# Enable multi-round vision understanding for enhanced accuracy
VLLM_ENABLE_MULTI_TURN = os.environ.get("VLLM_ENABLE_MULTI_TURN", "true").lower() == "true"

# Confidence threshold for vision understanding (0.0-1.0)
VLLM_CONFIDENCE_THRESHOLD = float(os.environ.get("VLLM_CONFIDENCE_THRESHOLD", "0.6"))

# ==================== Vision Understanding Configuration ====================

# ==================== Enhanced Audio Understanding ====================
# Multi-provider Audio Language Model support for advanced audio processing
# Supports transcription, question answering, and audio feature extraction

# Primary Audio Provider - supports: "openai_whisper", "whisper_os", "qwen_audio", "gpt4o_audio"
AUDIO_PROVIDER = os.environ.get("AUDIO_PROVIDER", "openai_whisper")

# OpenAI Whisper configuration (Commercial ASR)
WHISPER_OPENAI_API_KEY = os.environ.get("WHISPER_OPENAI_API_KEY") or OPENAI_API_KEY
WHISPER_OPENAI_BASE_URL = os.environ.get("WHISPER_OPENAI_BASE_URL") or OPENAI_BASE_URL
WHISPER_OPENAI_MODEL = os.environ.get("WHISPER_OPENAI_MODEL", "whisper-1")

# Open-Source Whisper configuration (Self-hosted or third-party)
WHISPER_OS_API_KEY = os.environ.get("WHISPER_OS_API_KEY") or WHISPER_API_KEY
WHISPER_OS_BASE_URL = os.environ.get("WHISPER_OS_BASE_URL") or WHISPER_BASE_URL
WHISPER_OS_MODEL = os.environ.get("WHISPER_OS_MODEL") or WHISPER_MODEL_NAME or "whisper-large-v3"

# Qwen-Audio configuration (Supports audio understanding + QA)
QWEN_AUDIO_API_KEY = os.environ.get("QWEN_AUDIO_API_KEY")
QWEN_AUDIO_BASE_URL = os.environ.get("QWEN_AUDIO_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_AUDIO_MODEL = os.environ.get("QWEN_AUDIO_MODEL", "qwen-audio-turbo")

# GPT-4o-audio configuration (Advanced audio understanding)
GPT4O_AUDIO_API_KEY = os.environ.get("GPT4O_AUDIO_API_KEY") or OPENAI_API_KEY
GPT4O_AUDIO_BASE_URL = os.environ.get("GPT4O_AUDIO_BASE_URL") or OPENAI_BASE_URL
GPT4O_AUDIO_MODEL = os.environ.get("GPT4O_AUDIO_MODEL", "gpt-4o-audio-preview")

# Enable multi-turn audio verification for enhanced accuracy
AUDIO_ENABLE_MULTI_TURN = os.environ.get("AUDIO_ENABLE_MULTI_TURN", "true").lower() == "true"

# Enable web search validation for audio transcriptions
AUDIO_ENABLE_WEB_SEARCH = os.environ.get("AUDIO_ENABLE_WEB_SEARCH", "false").lower() == "true"

# Confidence threshold for audio understanding (0.0-1.0)
AUDIO_CONFIDENCE_THRESHOLD = float(os.environ.get("AUDIO_CONFIDENCE_THRESHOLD", "0.6"))

# ==================== Audio Understanding Configuration ====================


# ==================== Enhanced Video Understanding ====================
# Multi-provider Video Language Model support for advanced video understanding
# Supports temporal analysis, action recognition, and scene understanding

# Primary Video Provider - supports: "gemini", "gpt4o_video", "qwen_video", "custom"
VIDEO_PROVIDER = os.environ.get("VIDEO_PROVIDER", "gemini")

# Google Gemini configuration (Primary - best video understanding as of 2025)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-exp")

# GPT-4o-video configuration (Backup - frame-based analysis)
GPT4O_VIDEO_API_KEY = os.environ.get("GPT4O_VIDEO_API_KEY") or OPENAI_API_KEY
GPT4O_VIDEO_BASE_URL = os.environ.get("GPT4O_VIDEO_BASE_URL") or OPENAI_BASE_URL
GPT4O_VIDEO_MODEL = os.environ.get("GPT4O_VIDEO_MODEL", "gpt-4o")

# Qwen-VL-Video configuration (Open-source option)
QWEN_VIDEO_API_KEY = os.environ.get("QWEN_VIDEO_API_KEY")
QWEN_VIDEO_BASE_URL = os.environ.get("QWEN_VIDEO_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_VIDEO_MODEL = os.environ.get("QWEN_VIDEO_MODEL", "qwen-vl-max")

# Custom video API configuration
CUSTOM_VIDEO_API_KEY = os.environ.get("CUSTOM_VIDEO_API_KEY")
CUSTOM_VIDEO_BASE_URL = os.environ.get("CUSTOM_VIDEO_BASE_URL")
CUSTOM_VIDEO_MODEL = os.environ.get("CUSTOM_VIDEO_MODEL")

# Enable multi-turn video verification for enhanced accuracy
VIDEO_ENABLE_MULTI_TURN = os.environ.get("VIDEO_ENABLE_MULTI_TURN", "true").lower() == "true"

# Enable web search validation for video analysis
VIDEO_ENABLE_WEB_SEARCH = os.environ.get("VIDEO_ENABLE_WEB_SEARCH", "false").lower() == "true"

# Confidence threshold for video understanding (0.0-1.0)
VIDEO_CONFIDENCE_THRESHOLD = float(os.environ.get("VIDEO_CONFIDENCE_THRESHOLD", "0.6"))

# Maximum keyframes to extract for analysis
VIDEO_MAX_KEYFRAMES = int(os.environ.get("VIDEO_MAX_KEYFRAMES", "10"))

# ==================== Video Understanding Configuration ====================


# MCP server configuration generation function
def create_mcp_server_parameters(cfg: DictConfig, agent_cfg: DictConfig):
    """Define and return MCP server configuration list"""
    configs = []

    if (
        agent_cfg.get("tools", None) is not None
        and "tool-google-search" in agent_cfg["tools"]
    ):
        if not SERPER_API_KEY:
            raise ValueError(
                "SERPER_API_KEY not set, tool-google-search will be unavailable."
            )

        configs.append(
            {
                "name": "tool-google-search",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=[
                        "-m",
                        "miroflow_tools.mcp_servers.searching_google_mcp_server",
                    ],
                    env={
                        "SERPER_API_KEY": SERPER_API_KEY,
                        "SERPER_BASE_URL": SERPER_BASE_URL,
                        "JINA_API_KEY": JINA_API_KEY,
                        "JINA_BASE_URL": JINA_BASE_URL,
                    },
                ),
            }
        )

    if (
        agent_cfg.get("tools", None) is not None
        and "tool-sougou-search" in agent_cfg["tools"]
    ):
        configs.append(
            {
                "name": "tool-sougou-search",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=[
                        "-m",
                        "miroflow_tools.mcp_servers.searching_sougou_mcp_server",
                    ],
                    env={
                        "TENCENTCLOUD_SECRET_ID": TENCENTCLOUD_SECRET_ID,
                        "TENCENTCLOUD_SECRET_KEY": TENCENTCLOUD_SECRET_KEY,
                        "JINA_API_KEY": JINA_API_KEY,
                        "JINA_BASE_URL": JINA_BASE_URL,
                    },
                ),
            }
        )

    if agent_cfg.get("tools", None) is not None and "tool-python" in agent_cfg["tools"]:
        configs.append(
            {
                "name": "tool-python",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=["-m", "miroflow_tools.mcp_servers.python_mcp_server"],
                    env={"E2B_API_KEY": E2B_API_KEY},
                ),
            }
        )

    if agent_cfg.get("tools", None) is not None and "tool-code" in agent_cfg["tools"]:
        configs.append(
            {
                "name": "tool-code",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=["-m", "miroflow_tools.mcp_servers.python_mcp_server"],
                    env={"E2B_API_KEY": E2B_API_KEY},
                ),
            }
        )

    if agent_cfg.get("tools", None) is not None and "tool-vqa" in agent_cfg["tools"]:
        configs.append(
            {
                "name": "tool-vqa",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=["-m", "miroflow_tools.mcp_servers.vision_mcp_server"],
                    env={
                        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
                        "ANTHROPIC_BASE_URL": ANTHROPIC_BASE_URL,
                    },
                ),
            }
        )

    if agent_cfg.get("tools", None) is not None and "tool-vqa-os" in agent_cfg["tools"]:
        configs.append(
            {
                "name": "tool-vqa-os",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=["-m", "miroflow_tools.mcp_servers.vision_mcp_server_os"],
                    env={
                        "VISION_API_KEY": VISION_API_KEY,
                        "VISION_BASE_URL": VISION_BASE_URL,
                        "VISION_MODEL_NAME": VISION_MODEL_NAME,
                    },
                ),
            }
        )

    # ==================== Enhanced VQA Tool (New) ====================
    # Advanced vision understanding with multi-round verification and cross-validation
    if agent_cfg.get("tools", None) is not None and "tool-vqa-enhanced" in agent_cfg["tools"]:
        vllm_env = {
            "VLLM_PROVIDER": VLLM_PROVIDER,
            "VLLM_ENABLE_MULTI_TURN": str(VLLM_ENABLE_MULTI_TURN),
            "VLLM_CONFIDENCE_THRESHOLD": str(VLLM_CONFIDENCE_THRESHOLD),
        }
        
        # Configure based on selected provider
        if VLLM_PROVIDER == "openai":
            vllm_env.update({
                "VLLM_API_KEY": VLLM_OPENAI_API_KEY,
                "VLLM_BASE_URL": VLLM_OPENAI_BASE_URL,
                "VLLM_MODEL": VLLM_OPENAI_MODEL,
            })
        elif VLLM_PROVIDER == "anthropic":
            vllm_env.update({
                "VLLM_API_KEY": VLLM_ANTHROPIC_API_KEY,
                "VLLM_BASE_URL": VLLM_ANTHROPIC_BASE_URL,
                "VLLM_MODEL": VLLM_ANTHROPIC_MODEL,
            })
        elif VLLM_PROVIDER == "qwen":
            vllm_env.update({
                "VLLM_API_KEY": VLLM_QWEN_API_KEY,
                "VLLM_BASE_URL": VLLM_QWEN_BASE_URL,
                "VLLM_MODEL": VLLM_QWEN_MODEL,
            })
        elif VLLM_PROVIDER == "custom":
            vllm_env.update({
                "VLLM_API_KEY": VLLM_CUSTOM_API_KEY,
                "VLLM_BASE_URL": VLLM_CUSTOM_BASE_URL,
                "VLLM_MODEL": VLLM_CUSTOM_MODEL,
            })
        
        # Add Google Search API for verification
        vllm_env.update({
            "SERPER_API_KEY": SERPER_API_KEY,
            "SERPER_BASE_URL": SERPER_BASE_URL,
            "JINA_API_KEY": JINA_API_KEY,
            "JINA_BASE_URL": JINA_BASE_URL,
        })
        
        configs.append(
            {
                "name": "tool-vqa-enhanced",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=["-m", "miroflow_tools.mcp_servers.enhanced_vqa_mcp_server"],
                    env=vllm_env,
                ),
            }
        )

    if (
        agent_cfg.get("tools", None) is not None
        and "tool-transcribe" in agent_cfg["tools"]
    ):
        configs.append(
            {
                "name": "tool-transcribe",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=["-m", "miroflow_tools.mcp_servers.audio_mcp_server"],
                    env={
                        "OPENAI_API_KEY": OPENAI_API_KEY,
                        "OPENAI_BASE_URL": OPENAI_BASE_URL,
                    },
                ),
            }
        )

    if (
        agent_cfg.get("tools", None) is not None
        and "tool-transcribe-os" in agent_cfg["tools"]
    ):
        configs.append(
            {
                "name": "tool-transcribe-os",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=["-m", "miroflow_tools.mcp_servers.audio_mcp_server_os"],
                    env={
                        "WHISPER_BASE_URL": WHISPER_BASE_URL,
                        "WHISPER_API_KEY": WHISPER_API_KEY,
                        "WHISPER_MODEL_NAME": WHISPER_MODEL_NAME,
                    },
                ),
            }
        )

    if (
        agent_cfg.get("tools", None) is not None
        and "tool-audio-enhanced" in agent_cfg["tools"]
    ):
        # Enhanced audio understanding with multi-provider support
        audio_env = {
            "AUDIO_PROVIDER": AUDIO_PROVIDER,
            "AUDIO_ENABLE_MULTI_TURN": str(AUDIO_ENABLE_MULTI_TURN).lower(),
            "AUDIO_ENABLE_WEB_SEARCH": str(AUDIO_ENABLE_WEB_SEARCH).lower(),
            "SERPER_API_KEY": SERPER_API_KEY or "",
        }

        # Add provider-specific environment variables
        if AUDIO_PROVIDER == "openai_whisper":
            audio_env.update({
                "WHISPER_OPENAI_API_KEY": WHISPER_OPENAI_API_KEY,
                "WHISPER_OPENAI_BASE_URL": WHISPER_OPENAI_BASE_URL,
                "WHISPER_OPENAI_MODEL": WHISPER_OPENAI_MODEL,
            })
        elif AUDIO_PROVIDER == "whisper_os":
            audio_env.update({
                "WHISPER_OS_API_KEY": WHISPER_OS_API_KEY or "",
                "WHISPER_OS_BASE_URL": WHISPER_OS_BASE_URL or "",
                "WHISPER_OS_MODEL": WHISPER_OS_MODEL,
            })
        elif AUDIO_PROVIDER == "qwen_audio":
            audio_env.update({
                "QWEN_AUDIO_API_KEY": QWEN_AUDIO_API_KEY or "",
                "QWEN_AUDIO_BASE_URL": QWEN_AUDIO_BASE_URL,
                "QWEN_AUDIO_MODEL": QWEN_AUDIO_MODEL,
            })
        elif AUDIO_PROVIDER == "gpt4o_audio":
            audio_env.update({
                "GPT4O_AUDIO_API_KEY": GPT4O_AUDIO_API_KEY,
                "GPT4O_AUDIO_BASE_URL": GPT4O_AUDIO_BASE_URL,
                "GPT4O_AUDIO_MODEL": GPT4O_AUDIO_MODEL,
            })

        configs.append(
            {
                "name": "tool-audio-enhanced",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=[
                        "-m",
                        "miroflow_tools.mcp_servers.enhanced_audio_mcp_server",
                    ],
                    env=audio_env,
                ),
            }
        )

    if (
        agent_cfg.get("tools", None) is not None
        and "tool-video-enhanced" in agent_cfg["tools"]
    ):
        # Enhanced video understanding with multi-provider support
        video_env = {
            "VIDEO_PROVIDER": VIDEO_PROVIDER,
            "VIDEO_ENABLE_MULTI_TURN": str(VIDEO_ENABLE_MULTI_TURN).lower(),
            "VIDEO_ENABLE_WEB_SEARCH": str(VIDEO_ENABLE_WEB_SEARCH).lower(),
            "VIDEO_MAX_KEYFRAMES": str(VIDEO_MAX_KEYFRAMES),
            "SERPER_API_KEY": SERPER_API_KEY or "",
        }

        # Add provider-specific environment variables
        if VIDEO_PROVIDER == "gemini":
            video_env.update({
                "GEMINI_API_KEY": GEMINI_API_KEY or "",
                "GEMINI_BASE_URL": GEMINI_BASE_URL,
                "GEMINI_MODEL": GEMINI_MODEL,
            })
        elif VIDEO_PROVIDER == "gpt4o_video":
            video_env.update({
                "GPT4O_VIDEO_API_KEY": GPT4O_VIDEO_API_KEY,
                "GPT4O_VIDEO_BASE_URL": GPT4O_VIDEO_BASE_URL,
                "GPT4O_VIDEO_MODEL": GPT4O_VIDEO_MODEL,
            })
        elif VIDEO_PROVIDER == "qwen_video":
            video_env.update({
                "QWEN_VIDEO_API_KEY": QWEN_VIDEO_API_KEY or "",
                "QWEN_VIDEO_BASE_URL": QWEN_VIDEO_BASE_URL,
                "QWEN_VIDEO_MODEL": QWEN_VIDEO_MODEL,
            })
        elif VIDEO_PROVIDER == "custom":
            video_env.update({
                "CUSTOM_VIDEO_API_KEY": CUSTOM_VIDEO_API_KEY or "",
                "CUSTOM_VIDEO_BASE_URL": CUSTOM_VIDEO_BASE_URL or "",
                "CUSTOM_VIDEO_MODEL": CUSTOM_VIDEO_MODEL or "",
            })

        configs.append(
            {
                "name": "tool-video-enhanced",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=[
                        "-m",
                        "miroflow_tools.mcp_servers.enhanced_video_mcp_server",
                    ],
                    env=video_env,
                ),
            }
        )

    if (
        agent_cfg.get("tools", None) is not None
        and "tool-image-search" in agent_cfg["tools"]
    ):
        configs.append(
            {
                "name": "tool-image-search",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=[
                        "-m",
                        "miroflow_tools.mcp_servers.image_search_mcp_server",
                    ],
                    env={
                        "JINA_API_KEY": JINA_API_KEY,
                        "JINA_BASE_URL": JINA_BASE_URL,
                        "VLLM_PROVIDER": VLLM_PROVIDER,
                        "VLLM_API_KEY": VLLM_OPENAI_API_KEY if VLLM_PROVIDER == "openai" else (
                            VLLM_ANTHROPIC_API_KEY if VLLM_PROVIDER == "anthropic" else (
                                VLLM_QWEN_API_KEY if VLLM_PROVIDER == "qwen" else 
                                VLLM_CUSTOM_API_KEY
                            )
                        ),
                        "VLLM_BASE_URL": VLLM_OPENAI_BASE_URL if VLLM_PROVIDER == "openai" else (
                            VLLM_ANTHROPIC_BASE_URL if VLLM_PROVIDER == "anthropic" else (
                                VLLM_QWEN_BASE_URL if VLLM_PROVIDER == "qwen" else 
                                VLLM_CUSTOM_BASE_URL
                            )
                        ),
                        "VLLM_MODEL": VLLM_OPENAI_MODEL if VLLM_PROVIDER == "openai" else (
                            VLLM_ANTHROPIC_MODEL if VLLM_PROVIDER == "anthropic" else (
                                VLLM_QWEN_MODEL if VLLM_PROVIDER == "qwen" else 
                                VLLM_CUSTOM_MODEL
                            )
                        ),
                    },
                ),
            }
        )

    if (
        agent_cfg.get("tools", None) is not None
        and "tool-reasoning" in agent_cfg["tools"]
    ):
        configs.append(
            {
                "name": "tool-reasoning",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=[
                        "-m",
                        "miroflow_tools.mcp_servers.reasoning_mcp_server",
                    ],
                    env={
                        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
                        "ANTHROPIC_BASE_URL": ANTHROPIC_BASE_URL,
                    },
                ),
            }
        )

    if (
        agent_cfg.get("tools", None) is not None
        and "tool-reasoning-os" in agent_cfg["tools"]
    ):
        configs.append(
            {
                "name": "tool-reasoning-os",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=[
                        "-m",
                        "miroflow_tools.mcp_servers.reasoning_mcp_server_os",
                    ],
                    env={
                        "REASONING_API_KEY": REASONING_API_KEY,
                        "REASONING_BASE_URL": REASONING_BASE_URL,
                        "REASONING_MODEL_NAME": REASONING_MODEL_NAME,
                    },
                ),
            }
        )

    # reader
    if agent_cfg.get("tools", None) is not None and "tool-reader" in agent_cfg["tools"]:
        configs.append(
            {
                "name": "tool-reader",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=["-m", "markitdown_mcp"],
                ),
            }
        )

    if (
        agent_cfg.get("tools", None) is not None
        and "tool-reading" in agent_cfg["tools"]
    ):
        configs.append(
            {
                "name": "tool-reading",
                "params": StdioServerParameters(
                    command=sys.executable,
                    args=["-m", "miroflow_tools.mcp_servers.reading_mcp_server"],
                ),
            }
        )

    blacklist = set()
    for black_list_item in agent_cfg.get("tool_blacklist", []):
        blacklist.add((black_list_item[0], black_list_item[1]))
    return configs, blacklist


def expose_sub_agents_as_tools(sub_agents_cfg: DictConfig):
    """Expose sub-agents as tools"""
    sub_agents_server_params = []
    for sub_agent in sub_agents_cfg.keys():
        if "agent-browsing" in sub_agent:
            sub_agents_server_params.append(
                dict(
                    name="agent-browsing",
                    tools=[
                        dict(
                            name="search_and_browse",
                            description="This tool is an agent that performs the subtask of searching and browsing the web for specific missing information and generating the desired answer. The subtask should be clearly defined, include relevant background, and focus on factual gaps. It does not perform vague or speculative subtasks. \nArgs: \n\tsubtask: the subtask to be performed. \nReturns: \n\tthe result of the subtask. ",
                            schema={
                                "type": "object",
                                "properties": {
                                    "subtask": {"title": "Subtask", "type": "string"}
                                },
                                "required": ["subtask"],
                                "title": "search_and_browseArguments",
                            },
                        )
                    ],
                )
            )
    return sub_agents_server_params


def get_env_info(cfg: DictConfig) -> dict:
    """Collect current configuration environment variable information for logging"""
    return {
        # LLM Configuration
        "llm_provider": cfg.llm.provider,
        "llm_base_url": cfg.llm.base_url,
        "llm_model_name": cfg.llm.model_name,
        "llm_temperature": cfg.llm.temperature,
        "llm_top_p": cfg.llm.top_p,
        "llm_min_p": cfg.llm.min_p,
        "llm_top_k": cfg.llm.top_k,
        "llm_max_tokens": cfg.llm.max_tokens,
        "llm_async_client": cfg.llm.async_client,
        "keep_tool_result": cfg.llm.keep_tool_result,
        # Agent Configuration
        "main_agent_max_turns": cfg.agent.main_agent.max_turns,
        **{
            f"sub_{sub_agent}_max_turns": cfg.agent.sub_agents[sub_agent].max_turns
            for sub_agent in cfg.agent.sub_agents
        },
        # API Keys (masked for security)
        "has_serper_api_key": bool(SERPER_API_KEY),
        "has_jina_api_key": bool(JINA_API_KEY),
        "has_anthropic_api_key": bool(ANTHROPIC_API_KEY),
        "has_openai_api_key": bool(OPENAI_API_KEY),
        "has_e2b_api_key": bool(E2B_API_KEY),
        "has_tencent_secret_id": bool(TENCENTCLOUD_SECRET_ID),
        "has_tencent_secret_key": bool(TENCENTCLOUD_SECRET_KEY),
        # Base URLs
        "openai_base_url": OPENAI_BASE_URL,
        "anthropic_base_url": ANTHROPIC_BASE_URL,
        "jina_base_url": JINA_BASE_URL,
        "serper_base_url": SERPER_BASE_URL,
        "whisper_base_url": WHISPER_BASE_URL,
        "vision_base_url": VISION_BASE_URL,
        "reasoning_base_url": REASONING_BASE_URL,
    }
