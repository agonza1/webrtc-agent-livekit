import os
import time
from prometheus_client import start_http_server, CollectorRegistry, multiprocess

# Set the same multiprocess directory used by all agent workers
os.environ['prometheus_multiproc_dir'] = '/tmp/prometheus_multiproc'

# Create the registry and collect from all processes
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

# Start the HTTP server (this exposes /metrics)
start_http_server(9100, addr="0.0.0.0", registry=registry)

print("Agent metrics aggregator running on port 9100")

# Keep the process alive
while True:
    time.sleep(10)
