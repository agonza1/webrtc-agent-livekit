FROM nginx:alpine

RUN rm /etc/nginx/conf.d/default.conf \
    && apk add --no-cache openssl

COPY nginx.conf /etc/nginx/nginx.conf

# Generate self-signed certs at container start if none are mounted
COPY entrypoint-generate-certs.sh /docker-entrypoint.d/00-generate-certs.sh
RUN chmod +x /docker-entrypoint.d/00-generate-certs.sh