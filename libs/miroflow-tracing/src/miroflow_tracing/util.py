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


from .setup import get_trace_provider


def time_iso() -> str:
    """Return the current time in ISO 8601 format."""
    return get_trace_provider().time_iso()


def gen_trace_id() -> str:
    """Generate a new trace ID."""
    return get_trace_provider().gen_trace_id()


def gen_span_id() -> str:
    """Generate a new span ID."""
    return get_trace_provider().gen_span_id()


def gen_group_id() -> str:
    """Generate a new group ID."""
    return get_trace_provider().gen_group_id()
