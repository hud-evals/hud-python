FROM python:3.12-alpine

COPY . /tests
RUN adduser -D -u 1001 verifier && mkdir -p /logs/verifier

USER root
ENTRYPOINT []
CMD ["sleep", "infinity"]
