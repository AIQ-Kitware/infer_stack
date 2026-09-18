#!/usr/bin/env bash
# Docker-only host checks for the admission plan (§7): V1 V2 V3 V5 V6 V10 V13 (+V8).
#
# Safe on a host with a live infer-stack: every check uses its own throwaway
# Compose project (isv-<random>) and removes it on exit. Nothing here touches the
# `infer-stack` project, its ledger, or its network. GPU checks need the NVIDIA
# runtime and a free GPU index (GPU=<index>, default 0); set GPU=none to skip them.
#
#   bash dev/tmp/host_verification_docker.sh 2>&1 | tee host-verification-docker.log
set -uo pipefail

GPU="${GPU:-0}"
IMAGE="${IMAGE:-busybox:latest}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.4.1-base-ubuntu22.04}"
TORCH_IMAGE="${TORCH_IMAGE:-}"          # optional, for V8 (e.g. pytorch/pytorch:latest)
PROJECT="isv-$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
WORK="$(mktemp -d)"
FAILS=0
trap 'docker compose -p "$PROJECT" -f "$WORK/c.yml" down -t 1 --remove-orphans >/dev/null 2>&1; docker network rm "${PROJECT}_fixed" >/dev/null 2>&1; rm -rf "$WORK"' EXIT

pass() { printf 'PASS %s\n' "$*"; }
fail() { printf 'FAIL %s\n' "$*"; FAILS=$((FAILS + 1)); }
note() { printf 'NOTE %s\n' "$*"; }
dc() { docker compose -p "$PROJECT" -f "$WORK/c.yml" "$@"; }
ids_of() { docker ps -a --no-trunc --filter "label=com.docker.compose.project=$PROJECT" \
             --filter "label=com.docker.compose.service=$1" --format '{{.ID}}'; }

write() { cat > "$WORK/c.yml"; }

docker pull -q "$IMAGE" >/dev/null

# -- V13: a changed label value recreates; an unchanged one keeps the container --------
write <<YML
services:
  s:
    image: $IMAGE
    command: [sleep, "3600"]
    labels: {infer-stack.fingerprint: "one"}
YML
dc up -d >/dev/null 2>&1
first=$(ids_of s)
dc up -d --no-deps s >/dev/null 2>&1
same=$(ids_of s)
sed -i 's/"one"/"two"/' "$WORK/c.yml"
dc up -d --no-deps s >/dev/null 2>&1
changed=$(ids_of s)
[[ "$first" == "$same" ]] && pass "V5 unchanged stanza with up -d --no-deps keeps the container" \
                           || fail "V5 unchanged stanza recreated the container"
[[ "$first" != "$changed" ]] && pass "V13 changed label value recreates the container" \
                             || fail "V13 changed label did not recreate"
dc down -t 1 >/dev/null 2>&1

# -- V2: project- and label-scoped ps -a lists every container, in every state ----------
write <<YML
services:
  run: {image: $IMAGE, command: [sleep, "3600"], labels: {infer-stack.deployment: grp-run}}
  made: {image: $IMAGE, command: [sleep, "3600"], labels: {infer-stack.deployment: grp-made}}
  quit: {image: $IMAGE, command: ["true"], labels: {infer-stack.deployment: grp-quit}}
YML
dc create >/dev/null 2>&1
dc start run quit >/dev/null 2>&1
sleep 2
listed=$(docker ps -a --no-trunc --filter "label=com.docker.compose.project=$PROJECT" --format '{{.State}}' | sort | tr '\n' ' ')
[[ "$listed" == *created* && "$listed" == *running* && "$listed" == *exited* ]] \
  && pass "V2 ps -a lists created, running and exited project containers ($listed)" \
  || fail "V2 listing incomplete: $listed"

# -- V6: crashed-container states under restart: unless-stopped ---------------------------
write <<YML
services:
  crash: {image: $IMAGE, command: [sh, -c, "sleep 1; exit 3"], restart: unless-stopped}
YML
dc up -d >/dev/null 2>&1
states=""
for _ in $(seq 1 12); do
  states+="$(docker inspect -f '{{.State.Status}}' "$(ids_of crash)") "
  sleep 0.5
done
note "V6 states observed for a crash-looping unless-stopped container: $(echo "$states" | tr ' ' '\n' | sort | uniq -c | tr '\n' ' ')"
dc down -t 1 >/dev/null 2>&1

# -- V10: does a stopped container keep a static ipv4_address reserved? --------------------
docker network create --subnet 10.231.77.0/28 "${PROJECT}_fixed" >/dev/null
docker run -d --name "${PROJECT}-a" --network "${PROJECT}_fixed" --ip 10.231.77.5 \
  --label "com.docker.compose.project=$PROJECT" "$IMAGE" sleep 3600 >/dev/null
docker stop -t 1 "${PROJECT}-a" >/dev/null
if docker run -d --name "${PROJECT}-b" --network "${PROJECT}_fixed" --ip 10.231.77.5 \
     --label "com.docker.compose.project=$PROJECT" "$IMAGE" sleep 3600 >/dev/null 2>&1; then
  note "V10 a STOPPED container does NOT keep its static address reserved (another container took it)"
  docker rm -f "${PROJECT}-b" >/dev/null
else
  note "V10 a stopped container keeps its static address reserved"
fi
docker rm -f "${PROJECT}-a" >/dev/null 2>&1
docker network rm "${PROJECT}_fixed" >/dev/null 2>&1

if [[ "$GPU" == none ]]; then
  note "GPU checks skipped (GPU=none)"
  echo "failures: $FAILS"; exit "$FAILS"
fi
docker pull -q "$CUDA_IMAGE" >/dev/null

# -- V1: DeviceRequests[].DeviceIDs holds the rendered GPU index ---------------------------
write <<YML
services:
  g:
    image: $CUDA_IMAGE
    command: [sleep, "3600"]
    deploy: {resources: {reservations: {devices: [{driver: nvidia, device_ids: ["$GPU"], capabilities: [gpu]}]}}}
YML
dc create >/dev/null 2>&1
ids=$(docker inspect -f '{{json .HostConfig.DeviceRequests}}' "$(ids_of g)")
[[ "$ids" == *"\"DeviceIDs\":[\"$GPU\"]"* ]] && pass "V1 DeviceIDs holds index $GPU ($ids)" \
                                             || fail "V1 DeviceRequests: $ids"

# -- V3: can `up --remove-orphans` start a service before an orphan frees its GPU? ---------
dc up -d >/dev/null 2>&1
cat > "$WORK/c.yml" <<YML
services:
  h:
    image: $CUDA_IMAGE
    command: [sh, -c, "nvidia-smi --query-compute-apps=pid --format=csv,noheader; sleep 3600"]
    deploy: {resources: {reservations: {devices: [{driver: nvidia, device_ids: ["$GPU"], capabilities: [gpu]}]}}}
YML
dc up -d --remove-orphans >/dev/null 2>&1 &
upid=$!
overlap=no
for _ in $(seq 1 40); do
  running=$(docker ps --filter "label=com.docker.compose.project=$PROJECT" --format '{{.Label "com.docker.compose.service"}}' | sort | tr '\n' ' ')
  [[ "$running" == *g* && "$running" == *h* ]] && overlap=yes
  sleep 0.25
done
wait "$upid"
note "V3 old and new GPU containers observed running at the same time: $overlap (yes => the barrier is necessary)"
dc down -t 1 >/dev/null 2>&1

# -- V8 (optional): a paused container keeps its GPU memory --------------------------------
if [[ -n "$TORCH_IMAGE" ]]; then
  write <<YML
services:
  mem:
    image: $TORCH_IMAGE
    command: [python, -c, "import torch,time; x=torch.empty(1<<30, device='cuda'); time.sleep(3600)"]
    deploy: {resources: {reservations: {devices: [{driver: nvidia, device_ids: ["$GPU"], capabilities: [gpu]}]}}}
YML
  dc up -d >/dev/null 2>&1
  sleep 20
  before=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
  docker pause "$(ids_of mem)" >/dev/null
  sleep 5
  after=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
  note "V8 GPU $GPU memory used: running ${before} MiB, paused ${after} MiB"
  docker unpause "$(ids_of mem)" >/dev/null
  dc down -t 1 >/dev/null 2>&1
else
  note "V8 skipped (set TORCH_IMAGE to an image with torch+CUDA)"
fi

echo "failures: $FAILS"
exit "$FAILS"
