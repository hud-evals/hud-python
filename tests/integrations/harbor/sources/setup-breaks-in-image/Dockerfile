FROM python:3.12-slim
COPY *.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl
WORKDIR /workspace
COPY . ./
ENV BREAK_SETUP=1
