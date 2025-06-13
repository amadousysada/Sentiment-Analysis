# OpenTelemetry / Azure Application Insights
from azure.monitor.opentelemetry.exporter import (AzureMonitorMetricExporter,
                                                  AzureMonitorTraceExporter)
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from settings import get_settings

conf = get_settings()

def get_metrics():
    # Azure instrumentation key or connection string
    insights_conn = conf.APPLICATIONINSIGHTS_CONNECTION_STRING

    # Resource identification
    resource = Resource(attributes={SERVICE_NAME: conf.SERVICE_NAME})

    # Setup Tracing
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    trace_exporter = AzureMonitorTraceExporter(connection_string=insights_conn)
    tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))

    # Setup Metrics
    exporter = AzureMonitorMetricExporter(connection_string=insights_conn)
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=500)
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)

    meter = metrics.get_meter(__name__)

    # Define a counter metric for predictions
    prediction_counter = meter.create_counter(
        "sentiment_api_prediction_count",
        description="Number of predictions made",
    )
    feedback_counter = meter.create_counter(
        "sentiment_api_feedback_count",
        description="Number of feedback events",
    )

    return prediction_counter, feedback_counter