FROM python:3.12-alpine
RUN apk add --no-cache curl
WORKDIR /app
EXPOSE 8080
ENTRYPOINT ["sh", "-c", "python3 -m http.server 8080 --directory /app >/tmp/main-http.log 2>&1 & exec \"$@\"", "--"]
