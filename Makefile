PYTHON ?= python
# Optional extra Helm values; the common case infers GPU profiles.
VALUES ?=

status:
	$(PYTHON) manage.py status

render:
	$(PYTHON) manage.py render

bootstrap-k3s:
	$(PYTHON) manage.py kube bootstrap --provider=k3s --apply

install-kubeai:
	$(PYTHON) manage.py kube install --apply $(if $(VALUES),--values "$(VALUES)",)

port-forward-kubeai:
	kubectl -n kubeai port-forward svc/kubeai 8000:80
