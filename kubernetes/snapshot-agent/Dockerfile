FROM ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 util-linux \
    && rm -rf /var/lib/apt/lists/*

COPY agent.py /opt/edo/agent.py
ENTRYPOINT ["python3", "/opt/edo/agent.py"]
