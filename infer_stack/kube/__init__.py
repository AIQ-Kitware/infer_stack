"""Distribution-neutral Kubernetes integration helpers.

The generic layer models only capabilities visible through Kubernetes APIs and
Helm.  Distribution provisioning is deliberately kept in leaf modules such as
:mod:`infer_stack.kube.k3s`; generic setup must not depend on one distribution.

This package is not a kubectl replacement: arbitrary cluster administration
stays with Kubernetes-native tools.
"""

from .manage import KubeManager, NodeLifecyclePlan, SetupPlan

__all__ = ['KubeManager', 'NodeLifecyclePlan', 'SetupPlan']
