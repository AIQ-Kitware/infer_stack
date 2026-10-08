"""Catalog-only mutations and restartable operations for one controller authority."""
from __future__ import annotations

import math
import time
from dataclasses import asdict

from ..leasing.cluster_placement import narrow
from ..leasing.models import EndpointRequest, LeaseState, Sharing
from ..leasing.profile import canonical_digest
from .operations import Operations


def public_operation(operation: dict) -> dict:
    """Expose progress/results, never the private launch/config snapshot."""
    return {key: operation[key] for key in ['id', 'principal', 'state', 'created_at', 'updated_at', 'result', 'error']}


class ControlService:
    """The HTTP tier only authenticates and transports these domain operations.

    One worker advances commands, with readiness outside the mutation lock.
    Committed leases have deterministic operation-scoped IDs so recovery after
    a commit but before progress publication cannot acquire a duplicate lease.
    """

    def __init__(self, controller, catalog, *, clock=time.time, usage_path=None):
        self.controller, self.catalog, self.clock = controller, catalog, clock
        self.operations = Operations(controller.ledger.store, clock=clock)
        self._last_room = float('-inf')
        self.usage_path = usage_path

    def usage(self) -> list[dict]:
        from pathlib import Path

        from ..accounting import UsageStore, energy

        if self.usage_path is None or not Path(self.usage_path).exists():
            return []
        store = UsageStore(self.usage_path, readonly=True)
        try:
            return [{'facts': asdict(record), 'energy': energy(record)} for record in store.records()]
        finally:
            store.close()

    def nodes(self):
        observe = getattr(self.controller.backend, 'node_inventory', None)
        if observe is None:
            raise NotImplementedError('This backend has no remote node inventory')
        return observe()

    def maintain(self):
        """Reconcile using approved inputs, including first publication."""
        with self.controller.catalog_snapshot([self.catalog.source]):
            return self.controller.reconcile()

    def acquire(self, principal: str, key: str, payload: dict, *, identity=None) -> dict:
        return self._submit_endpoints(principal, key, payload, identity=identity, action='acquire')

    def access(self, principal: str, key: str, payload: dict, *, identity=None) -> dict:
        return self._submit_endpoints(principal, key, payload, identity=identity, action='access')

    def _submit_endpoints(self, principal, key, payload, *, identity, action):
        allowed = {'endpoints', 'ttl_seconds', 'timeout', 'dedicated', 'node', 'pool', 'queue'}
        if set(payload) - allowed:
            raise ValueError('Acquire accepts approved endpoint names and policy options only')
        names = payload.get('endpoints')
        if not isinstance(names, list) or not names or len(names) > 128 or any(not isinstance(n, str) or not n or len(n) > 253 for n in names):
            raise ValueError('endpoints must be a nonempty list of catalog names')
        ttl, timeout = payload.get('ttl_seconds'), payload.get('timeout', 600)
        for value in [ttl, timeout]:
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                raise ValueError('TTL and timeout must be finite positive seconds')
        if timeout is None or timeout > 86400:
            raise ValueError('timeout must be at most 86400 seconds')
        if any(not isinstance(payload.get(flag, False), bool) for flag in ['dedicated', 'queue']):
            raise ValueError('dedicated and queue must be boolean')
        existing = self.operations.find(principal, key)
        if existing:
            if existing['request']['action'] != action or existing['request']['client'] != payload:
                raise ValueError('Idempotency key already identifies a different request')
            return public_operation(existing)
        sharing = Sharing.DEDICATED if payload.get('dedicated') else None
        catalog = self.catalog
        endpoints = catalog.resolve(names)
        external = [e.semantic() for e in endpoints if not e.managed]
        if action == 'acquire':
            requests = catalog.resolve_requests(names, sharing=sharing)
        else:
            requests = [e.to_request(sharing_override=sharing) for e in endpoints if e.managed]
            if external and (payload.get('node') is not None or payload.get('pool') is not None):
                raise ValueError('Cluster placement cannot constrain an external endpoint')
        requests = narrow(requests, node=payload.get('node'), pool=payload.get('pool'))
        if identity is not None:
            if not identity.can_lease:
                raise PermissionError('Identity cannot acquire leases')
            if identity.allowed_endpoints is not None and any(e.alias not in identity.allowed_endpoints for e in endpoints):
                raise PermissionError('Requested endpoint is outside this identity policy')
            if requests and identity.max_ttl_seconds is not None and (ttl is None or ttl > identity.max_ttl_seconds):
                raise PermissionError('Lease requires a finite TTL within this identity policy')
        def admit(conn):
            if not requests or identity is None or identity.max_active_leases is None:
                return
            held = conn.execute("SELECT id FROM leases WHERE owner=? AND state='active' AND (expires_at IS NULL OR expires_at>?)", (principal, self.clock())).fetchall()
            active = {row['id'] for row in held}
            queued = [self.operations.decode(row) for row in conn.execute("SELECT * FROM control_operations WHERE principal=? AND state IN ('queued','running')", (principal,))]
            pending = sum(op['request']['action'] in {'acquire', 'access'} and bool(op['request'].get('requests'))
                          and 'lease-' + op['id'].removeprefix('op-') not in active for op in queued)
            if len(active) + pending >= identity.max_active_leases:
                raise PermissionError('Active and pending lease quota is exhausted')
        request = {'action': action, 'client': payload, 'requests': [asdict(r) for r in requests],
                   'aliases': [e.alias for e in endpoints], 'external': external,
                   'catalog_sources': [catalog.source],
                   'catalog_revision': canonical_digest(catalog.source)}
        return public_operation(self.operations.submit(principal, key, request, admit=admit))

    def release(self, principal: str, key: str, lease_id: str, *, operator=False, evict=False) -> dict:
        lease = self.controller.ledger.get_lease(lease_id)
        if lease is None:
            raise KeyError(lease_id)
        if lease.owner != principal and not operator:
            raise PermissionError('Only the lease owner or an operator can release it')
        return public_operation(self.operations.submit(principal, key,
                                {'action': 'release', 'client': {'lease_id': lease_id, 'evict': evict}}))

    def wait(self, principal: str, key: str, payload: dict, *, operator=False) -> dict:
        if set(payload) - {'lease_id', 'endpoints', 'timeout'}:
            raise ValueError('Wait accepts lease_id, endpoints and timeout only')
        lease = self.controller.ledger.get_lease(payload.get('lease_id'))
        if lease is None:
            raise KeyError('Unknown lease')
        if lease.owner != principal and not operator:
            raise PermissionError('Only the lease owner or an operator can wait on it')
        timeout = payload.get('timeout', 600)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 86400:
            raise ValueError('timeout must be finite and within 0..86400 seconds')
        endpoints = payload.get('endpoints', lease.endpoints)
        if not isinstance(endpoints, list) or not endpoints or any(ep not in lease.endpoints for ep in endpoints):
            raise ValueError('Wait endpoints must be held by this lease')
        return public_operation(self.operations.submit(principal, key, {'action': 'wait', 'client': payload}))

    def renew(self, principal: str, key: str, payload: dict, *, operator=False) -> dict:
        if set(payload) - {'lease_id', 'ttl_seconds'}:
            raise ValueError('Renew accepts lease_id and ttl_seconds only')
        lease = self.controller.ledger.get_lease(payload.get('lease_id'))
        if lease is None:
            raise KeyError('Unknown lease')
        if lease.owner != principal and not operator:
            raise PermissionError('Only the lease owner or an operator can renew it')
        ttl = payload.get('ttl_seconds')
        if ttl is not None and (isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or ttl <= 0):
            raise ValueError('TTL must be finite positive seconds or null')
        return public_operation(self.operations.submit(principal, key, {'action': 'renew', 'client': payload}))

    def administer(self, principal: str, key: str, action: str, payload: dict) -> dict:
        """Small operator vocabulary; never accepts backend commands or specs."""
        allowed = {'reconcile': set(), 'gc': {'evict_idle'}, 'evict': {'deployments'}}
        if action not in allowed or set(payload) - allowed[action]:
            raise ValueError('Unsupported operator operation or options')
        if action == 'gc' and not isinstance(payload.get('evict_idle', False), bool):
            raise ValueError('evict_idle must be boolean')
        if action == 'evict':
            ids = payload.get('deployments')
            if ids is not None and (not isinstance(ids, list) or any(not isinstance(gid, str) for gid in ids)):
                raise ValueError('deployments must be a list of IDs or null for all idle deployments')
        return public_operation(self.operations.submit(principal, key, {'action': action, 'client': payload}))

    def status(self) -> dict:
        leases, deployments = self.controller.ledger.status(virtual_expiry=True)
        observed = {}
        observe = getattr(self.controller.backend, 'lifecycle', None)
        if observe is not None:
            try:
                observed = observe(deployments)
            except Exception:
                observed = {g.id: {'phase': 'UNKNOWN', 'next_action': 'Restore backend observation access'} for g in deployments}
        return {'catalog_revision': canonical_digest(self.catalog.source),
                'leases': [{'id': le.id, 'owner': le.owner, 'state': le.state,
                            'endpoints': le.endpoints, 'deployment_ids': le.deployment_ids,
                            'expires_at': le.expires_at} for le in leases],
                'deployments': [{'id': g.id, 'engine': g.engine, 'state': g.state,
                                 'demand': g.demand, 'endpoints': sorted(g.served),
                                 'lifecycle': observed.get(g.id)} for g in deployments],
                'operations': [public_operation(op) for op in self.operations.all()]}

    def step(self) -> None:
        """Advance one queued mutation, then probe existing converging operations."""
        pending = sorted([op for op in self.operations.all() if op['state'] in {'queued', 'running'}],
                         key=lambda op: (op['updated_at'], op['id']))
        if pending:
            op = pending[0]
            self.operations.transition(op['id'], 'running')
            try:
                if op['request']['action'] == 'release':
                    self.controller.release_leases([op['request']['client']['lease_id']], evict=op['request']['client'].get('evict', False))
                    self.operations.transition(op['id'], 'succeeded', result={'lease_id': op['request']['client']['lease_id']})
                elif op['request']['action'] == 'renew':
                    payload = op['request']['client']
                    ttl = payload.get('ttl_seconds')
                    # Anchor retries to acceptance, so a lost response cannot
                    # extend the protection window a second time.
                    remaining = None if ttl is None else op['created_at'] + ttl - self.clock()
                    if remaining is not None and remaining <= 0:
                        raise ValueError('Requested renewal window has elapsed')
                    renewed = self.controller.renew(payload['lease_id'], ttl_seconds=remaining)
                    if renewed.lease is None:
                        raise ValueError('Lease is no longer active; acquire again')
                    self.operations.transition(op['id'], 'succeeded', result={'lease_id': renewed.lease.id, 'expires_at': renewed.lease.expires_at})
                elif op['request']['action'] in {'reconcile', 'gc', 'evict'}:
                    action, payload = op['request']['action'], op['request']['client']
                    if action == 'reconcile':
                        with self.controller.catalog_snapshot([self.catalog.source]):
                            rec = self.controller.apply_now()
                        result = {'publication_pending': rec.publication_pending}
                    elif action == 'gc':
                        outcome = self.controller.gc(evict_idle=payload.get('evict_idle', False))
                        result = {'expired_leases': outcome.expired_lease_ids, 'evicted': outcome.evicted_deployment_ids}
                    else:
                        outcome = self.controller.evict(payload.get('deployments'))
                        result = {'evicted': outcome.evicted_deployment_ids}
                    self.operations.transition(op['id'], 'succeeded', result=result)
                else:
                    self._admit(op)
            except Exception as ex:
                from ..leasing.backend import PlacementError
                if (isinstance(ex, PlacementError) and ex.capacity and op['request']['client'].get('queue')
                        and self.clock() < op['created_at'] + op['request']['client'].get('timeout', 600)):
                    # Rotate blocked admission behind release/renew commands;
                    # waiting inside the worker would prevent capacity release.
                    self.operations.transition(op['id'], 'queued')
                else:
                    self._fail(op, ex)
        for op in self.operations.all():
            if op['state'] == 'converging':
                self._probe(op)

    def _fail(self, op, ex):
        from .._log import logger
        logger.exception('Control operation {} failed', op['id'])
        # Acquisition can fail after its durable lease commit. Publish a
        # terminal failure only once cleanup is durable; if cleanup itself
        # fails leave the operation running so restart/retry can recover.
        if op['request']['action'] in {'acquire', 'access'}:
            lease_id = 'lease-' + op['id'].removeprefix('op-')
            lease = self.controller.ledger.get_lease(lease_id)
            if lease is not None and lease.state == LeaseState.ACTIVE:
                self.controller.release(lease_id)
        self.operations.transition(op['id'], 'failed', error={'code': type(ex).__name__, 'message': 'Controller could not complete the operation; inspect server diagnostics'})

    def _admit(self, operation: dict) -> None:
        from ..leasing.endpoints import (
            ExternalTarget,
            ManagedTarget,
            ResolvedEndpoint,
        )

        prospective_id = 'lease-' + operation['id'].removeprefix('op-')
        request = operation['request']
        client = request['client']
        waiting, accessing = request['action'] == 'wait', request['action'] == 'access'
        existing = self.controller.ledger.get_lease(client['lease_id'] if waiting else prospective_id)
        if waiting and existing is None:
            raise KeyError('Unknown lease')
        if existing is None and self.clock() >= operation['created_at'] + client.get('timeout', 600):
            raise TimeoutError('Admission deadline elapsed before publication')
        published = True
        with self.controller.catalog_snapshot(request.get('catalog_sources')):
            if existing is None:
                requests = [EndpointRequest(**r) for r in request['requests']]
                if accessing:
                    endpoints = {r.endpoint: ResolvedEndpoint(r.endpoint, r.served.get('protocol', 'chat'), ManagedTarget(r)) for r in requests}
                    for external in request['external']:
                        target = {k: v for k, v in external['target'].items() if k != 'kind'}
                        endpoints[external['alias']] = ResolvedEndpoint(external['alias'], external['protocol'], ExternalTarget(**target))
                    outcome = self.controller.access(operation['principal'], [endpoints[alias] for alias in request['aliases']],
                                                     ttl_seconds=client.get('ttl_seconds'), wait=False, lease_id=prospective_id)
                    lease, published = outcome.lease, outcome.published
                else:
                    outcome = self.controller.acquire(operation['principal'], requests,
                                ttl_seconds=client.get('ttl_seconds'), wait=False, lease_id=prospective_id)
                    lease = outcome.lease
            else:
                lease = existing
                if accessing and request['external']:
                    published = not self.controller.publish_endpoints().publication_pending
                else:
                    self.controller.reconcile()
        if (lease is None and not (accessing and not request['requests'])) or (lease is not None and lease.state != LeaseState.ACTIVE):
            raise ValueError('Operation lease is no longer active')
        aliases = request['aliases'] if accessing else lease.endpoints
        info = self.controller.backend.connection_info()
        result = {'lease_id': lease.id if lease is not None else None, 'endpoints': aliases,
                  'external': [e['alias'] for e in request.get('external', [])], 'published': published,
                  'base_url': info.base_url if info else None,
                  'api_key_env': 'INFER_STACK_INFERENCE_KEY',
                  'models': self.controller.backend.request_names(aliases),
                  'catalog_revision': request.get('catalog_revision')}
        self.operations.transition(operation['id'], 'converging', result=result)

    def _probe(self, operation: dict) -> None:
        request = operation['request']
        lease_id = operation['result']['lease_id']
        lease = self.controller.ledger.get_lease(lease_id) if lease_id is not None else None
        external_only = request['action'] == 'access' and not request['requests']
        if not external_only and (lease is None or lease.state != LeaseState.ACTIVE or (lease.expires_at is not None and lease.expires_at <= self.clock())):
            self.operations.transition(operation['id'], 'superseded', error={'code': 'LeaseUnavailable', 'message': 'Lease was released or expired'})
            return
        waiting = request['action'] == 'wait'
        managed_endpoints = lease.endpoints if lease is not None else []
        endpoints = request['client'].get('endpoints', managed_endpoints) if waiting else managed_endpoints
        deployments = [self.controller.ledger.get_deployment(gid) for gid in set(lease.deployment_ids)] if lease is not None else []
        gateway_ready = True
        try:
            if request.get('external'):
                from ..leasing.profile import CatalogUnion

                published = CatalogUnion.from_sources((self.controller.ledger.profile() or {}).get('catalogs') or [])
                if any(e['alias'] not in published.endpoints or published.resolve_endpoint(e['alias']).semantic() != e for e in request['external']):
                    if lease is not None:
                        self.controller.release(lease.id)
                    self.operations.transition(operation['id'], 'superseded', error={'code': 'EndpointChanged', 'message': 'An external endpoint was changed or unpublished before access completed'})
                    return
                front = self.controller.backend.front_door()
                gateway_ready = (front.gateway_accepts(front.master_key(), wait=0) if front is not None and not self.controller.ledger.publication_pending() else None)
            probes = [self.controller.backend.probe_ready(g, ep)
                      for g in deployments if g is not None for ep in endpoints if ep in g.served]
        except Exception:
            probes = []  # retry observation, but still enforce the deadline
            gateway_ready = None
        if gateway_ready is True and len(probes) == len(endpoints) and all(p.ready for p in probes):
            self.operations.transition(operation['id'], 'succeeded', result={**operation['result'], 'published': True})
        elif gateway_ready is False or any(p.fatal for p in probes) or self.clock() >= operation['created_at'] + request['client'].get('timeout', 600):
            if not waiting and lease is not None:
                self.controller.release(lease.id)
            suffix = '; lease retained' if waiting else '; lease released' if lease is not None else '; no lease held'
            self.operations.transition(operation['id'], 'failed', error={'code': 'ReadinessFailed', 'message': 'Endpoint readiness failed' + suffix})
        elif any(p.needs_room for p in probes):
            from ..leasing.controller import ROOM_COOLDOWN_S
            if self.clock() - self._last_room >= ROOM_COOLDOWN_S:
                if self.controller._make_room([g for g in deployments if g is not None]):
                    self._last_room = self.clock()
