# The fixture environments as containers: this checkout's SDK and one fixture env,
# served the way a deployed environment image serves its own.
FROM python:3.12-slim
COPY *.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
COPY envs/ /hud/
ARG FIXTURE
ENV FIXTURE=${FIXTURE}
EXPOSE 8765
CMD ["sh", "-c", "exec hud serve \"/hud/$FIXTURE.py\" --host 0.0.0.0 --port 8765"]
