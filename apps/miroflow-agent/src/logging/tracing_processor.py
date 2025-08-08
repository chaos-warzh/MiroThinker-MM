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


from collections import defaultdict
from typing import Any, Dict, List, Optional

from miroflow_tracing import (
    Span,
    Trace,
    TracingProcessor,
    add_trace_processor,
    set_tracing_disabled,
    util,
)


class FileTraceProcessor(TracingProcessor):
    """A tracing processor that stores completed traces in memory."""

    def __init__(self):
        self.traces_by_task_id: Dict[str, List[Trace]] = defaultdict(list)
        self.spans_by_trace_id: Dict[str, List[Span]] = defaultdict(list)

    def on_trace_start(self, trace: "Trace") -> None:
        setattr(trace, "started_at", util.time_iso())

    def on_trace_end(self, trace: "Trace") -> None:
        setattr(trace, "ended_at", util.time_iso())
        setattr(trace, "spans", self.spans_by_trace_id.pop(trace.trace_id, []))
        if trace.trace_id:
            self.traces_by_task_id[trace.trace_id].append(trace)

    def on_span_start(self, span: "Span[Any]") -> None:
        pass

    def on_span_end(self, span: "Span[Any]") -> None:
        self.spans_by_trace_id[span.trace_id].append(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass

    def get_and_clear_traces(self, task_id: Optional[str] = None) -> List[Trace]:
        """Returns the stored traces for a specific task and clears them."""
        if not task_id:
            return []
        traces = self.traces_by_task_id.pop(task_id, [])
        return traces


_file_trace_processor_instance: Optional[FileTraceProcessor] = None


def setup_file_trace_processor() -> FileTraceProcessor:
    """
    Sets up a custom trace processor for the benchmark and returns it.
    Ensures that only one instance of the processor is created and used globally.
    """
    global _file_trace_processor_instance
    set_tracing_disabled(False)
    if _file_trace_processor_instance is None:
        _file_trace_processor_instance = FileTraceProcessor()
        add_trace_processor(_file_trace_processor_instance)
    return _file_trace_processor_instance
