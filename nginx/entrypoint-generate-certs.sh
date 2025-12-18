#!/bin/sh
set -e

CERT_DIR="/etc/certs"
FULLCHAIN="$CERT_DIR/fullchain.pem"
PRIVKEY="$CERT_DIR/privkey.pem"

mkdir -p "$CERT_DIR"

if [ ! -s "$FULLCHAIN" ] || [ ! -s "$PRIVKEY" ]; then
  echo "[nginx] No TLS certs found at $CERT_DIR, generating self-signed certs..."
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$PRIVKEY" -out "$FULLCHAIN" -days 365 \
    -subj "/C=US/ST=Local/L=Local/O=Dev/CN=localhost"
  chmod 644 "$FULLCHAIN" "$PRIVKEY"
  echo "[nginx] Self-signed certs generated."
else
  echo "[nginx] Found existing TLS certs at $CERT_DIR."
fi

exit 0


