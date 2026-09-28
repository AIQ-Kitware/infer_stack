"""Kubernetes integration helpers used by ``infer-stack kube``.

This package deliberately manages only the Kubernetes capabilities infer-stack
needs.  It is not a kubectl replacement: arbitrary cluster administration stays
with Kubernetes-native tools.
"""

from .manage import KubeManager, SetupPlan

__all__ = ['KubeManager', 'SetupPlan']
