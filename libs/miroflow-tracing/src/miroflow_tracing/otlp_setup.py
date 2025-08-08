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

from .otlp_exporter import FanoutExporter, NoopExporter, OtelExporter
from .processors import BatchTraceProcessor, ConsoleSpanExporter
from .provider import DefaultTraceProvider, TraceProvider
from .setup import set_trace_provider

# _old_provider = get_trace_provider()


def bootstrap_silent_trace_provider() -> TraceProvider:
    noop_exporter = NoopExporter()
    batch_processor = BatchTraceProcessor(exporter=noop_exporter)
    provider = DefaultTraceProvider()
    provider.set_processors([batch_processor])
    set_trace_provider(provider)
    return provider


def bootstrap_otlp_trace_provider() -> TraceProvider:
    console_exporter = ConsoleSpanExporter()
    otel_exporter = OtelExporter(endpoint="http://localhost:4318/v1/traces")
    fanout_exporter = FanoutExporter(exporters=[console_exporter, otel_exporter])
    batch_processor = BatchTraceProcessor(exporter=fanout_exporter)
    provider = DefaultTraceProvider()
    provider.set_processors([batch_processor])
    # _old_provider = get_trace_provider()
    set_trace_provider(provider)
    return provider


# def shutdown_otlp_trace_provider():
#     """
#     Shutdown the OTLP trace provider and restore the old provider.
#     """
#     provider = get_trace_provider()
#     if isinstance(provider, DefaultTraceProvider):
#         provider.shutdown()
#     set_trace_provider(_old_provider)
#     return _old_provider
