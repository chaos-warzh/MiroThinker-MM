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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .provider import TraceProvider

GLOBAL_TRACE_PROVIDER: TraceProvider | None = None


def set_trace_provider(provider: TraceProvider) -> None:
    """Set the global trace provider used by tracing utilities."""
    global GLOBAL_TRACE_PROVIDER
    GLOBAL_TRACE_PROVIDER = provider


def get_trace_provider() -> TraceProvider:
    """Get the global trace provider used by tracing utilities."""
    if GLOBAL_TRACE_PROVIDER is None:
        raise RuntimeError("Trace provider not set")
    return GLOBAL_TRACE_PROVIDER
