#!/usr/bin/env bash
# A second k3s node on this host, in a Docker container: enough to exercise
# anything that depends on the cluster having more than one node (node
# labels, node selectors, where a pod lands) without a second machine.
#
#   dev/k3s_agent_container.sh up   [NAME]    # join, taint, print the node
#   dev/k3s_agent_container.sh down [NAME]    # drain it out of the cluster
#
# The node is tainted `infer-stack.test/simulated=true:NoSchedule`, so only
# pods that tolerate it land there: an ordinary Model never does (its image
# would be pulled inside the container's own containerd). Give a test
# resource profile that toleration to put a Model on it.
#
# Needs: this host running the k3s server (scripts/bootstrap_k3s.sh), sudo
# to read the node token, and Docker. The image tag follows the server's.
set -euo pipefail

ACTION="${1:-up}"
NAME="${2:-k3s-agent-b}"
CONTAINER="infer-stack-${NAME}"

case "$ACTION" in
  up)
    version="$(k3s --version | awk 'NR==1 {print $3}')"     # v1.36.4+k3s1
    image="rancher/k3s:${version/+/-}"
    server_ip="$(ip -4 route get 1.1.1.1 | awk '{for (i=1;i<NF;i++) if ($i=="src") print $(i+1)}')"
    token="$(sudo cat /var/lib/rancher/k3s/server/node-token)"
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    docker run -d --name "$CONTAINER" --privileged --hostname "$NAME" \
      -e K3S_URL="https://${server_ip}:6443" -e K3S_TOKEN="$token" \
      --tmpfs /run --tmpfs /var/run "$image" agent --node-name "$NAME" >/dev/null
    for _ in $(seq 60); do
      kubectl get node "$NAME" >/dev/null 2>&1 && break
      sleep 2
    done
    kubectl wait --for=condition=Ready "node/$NAME" --timeout=120s >/dev/null
    kubectl taint node "$NAME" infer-stack.test/simulated=true:NoSchedule --overwrite >/dev/null
    kubectl get node "$NAME" -o wide
    ;;
  down)
    kubectl delete node "$NAME" --ignore-not-found >/dev/null
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    echo "removed $NAME"
    ;;
  *)
    echo "usage: $0 up|down [NAME]" >&2
    exit 2
    ;;
esac
