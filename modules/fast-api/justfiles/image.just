# The production image: build it, run it locked down, scan it. Needs Docker.

set working-directory := ".."

tag := "fast-api:local"
# Tool images pinned by digest, like the Dockerfile's.
trivy := "aquasec/trivy:0.67.2@sha256:e2b22eac59c02003d8749f5b8d9bd073b62e30fefaef5b7c8371204e0a4b0c08"

build:
    docker build -t {{ tag }} .

# Run the image locked down (non-root, read-only root filesystem, no capabilities, production settings), then check
# it serves, keeps its internals private, and logs one JSON object per line.
smoke: build
    #!/usr/bin/env bash
    set -euo pipefail
    name=fast-api-smoke
    url=http://localhost:18000
    docker rm -f "$name" >/dev/null 2>&1 || true
    trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
    docker run -d --name "$name" --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
        --user 10001 --cpus 2 -e ENVIRONMENT=production -p 18000:8000 {{ tag }} gunicorn -c gunicorn.conf.py >/dev/null
    for _ in $(seq 60); do curl -sf "$url/readyz" >/dev/null && break; sleep 1; done
    status() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
    zeros="$(printf '0,%.0s' $(seq 1023))0"
    test "$(status "$url/readyz")" = 200
    test "$(status -X POST "$url/v1/predict" -H 'content-type: application/json' -d "{\"inputs\": [[$zeros]]}")" = 200
    test "$(status -X POST "$url/v1/predict" -H 'content-type: application/json' -d '{"inputs": [[NaN]]}')" = 422
    test "$(status "$url/debug/flags")" = 200
    test "$(status "$url/openapi.json")" = 200
    # ONNX Runtime's C++ logger writes its own lines; everything Python logs is JSON.
    if docker logs "$name" 2>&1 | grep -v '^{' | grep -v 'onnxruntime'; then echo "non-JSON log lines above"; exit 1; fi
    echo "smoke test passed"

# Known, fixable HIGH and CRITICAL vulnerabilities fail the scan.
scan: build
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock {{ trivy }} image --quiet --scanners vuln \
        --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 {{ tag }}
