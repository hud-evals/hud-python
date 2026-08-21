FROM python:3.12-alpine

ENV VERIFIER_PRECEDENCE=verifier-image
COPY . /tests
RUN apk add --no-cache bash \
    && addgroup -g 1001 verifier \
    && adduser -D -u 1001 -G verifier verifier \
    && mkdir -p /app /logs/verifier /opt/result \
    && echo base > /opt/result/base.txt \
    && touch /home/verifier/owned \
    && chown verifier:verifier /home/verifier/owned

WORKDIR /home/verifier
USER verifier
ENTRYPOINT []
CMD ["sleep", "infinity"]
