FROM python:3.12-alpine

COPY . /tests
RUN addgroup -g 1001 verifier \
    && adduser -D -u 1001 -G verifier verifier \
    && chmod +x /tests/test.sh \
    && mkdir -p /app /logs/verifier

USER verifier
ENTRYPOINT []
CMD ["sleep", "infinity"]
