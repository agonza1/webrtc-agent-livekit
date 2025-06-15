FROM python:3.11-slim

RUN pip install prometheus_client

COPY agent-metrics.py /agent-metrics.py

CMD ["python", "/agent-metrics.py"]