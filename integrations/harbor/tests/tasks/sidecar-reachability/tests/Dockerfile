FROM python:3.12-alpine

COPY . /tests
RUN addgroup -g 1001 verifier \
    && adduser -D -u 1001 -G verifier verifier \
    && chmod 755 /tests/test.sh \
    && mkdir -p /app /logs/verifier \
    && touch /home/verifier/owned \
    && chown verifier:verifier /home/verifier/owned

WORKDIR /home/verifier
USER verifier
ENTRYPOINT []
CMD ["sleep", "infinity"]
