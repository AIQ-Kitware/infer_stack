PYTHON ?= python
# The KubeAI chart values: your resourceProfiles (see docs/kubeai-backend.md).
VALUES ?= kubeai-values.yaml

status:
	$(PYTHON) manage.py status

render:
	$(PYTHON) manage.py render

bootstrap-k3s:
	bash scripts/bootstrap_k3s.sh

install-kubeai:
	bash scripts/install_kubeai.sh $(VALUES) kubeai

port-forward-kubeai:
	kubectl -n kubeai port-forward svc/kubeai 8000:80
