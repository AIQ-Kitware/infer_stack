"""Reloadable operator token or named identities with scoped capabilities."""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Identity:
    principal: str
    role: str = 'operator'
    allowed_endpoints: tuple[str, ...] | None = None
    max_active_leases: int | None = None
    max_ttl_seconds: float | None = None

    @property
    def operator(self):
        return self.role in {'operator', 'catalog-admin'}

    @property
    def can_lease(self):
        return self.operator or self.role == 'lease-user'


def identities(text: str) -> list[tuple[str, Identity]]:
    rows = json.loads(text)
    if not isinstance(rows, dict) or set(rows) != {'identities'} or not isinstance(rows['identities'], list):
        raise ValueError('Identity file requires an identities list')
    result, principals, hashes = [], set(), set()
    for row in rows['identities']:
        allowed = {'principal', 'role', 'token_sha256', 'allowed_endpoints', 'max_active_leases', 'max_ttl_seconds'}
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError('Invalid identity fields')
        principal, role, digest = row.get('principal'), row.get('role'), row.get('token_sha256')
        if not isinstance(principal, str) or not principal or len(principal) > 200 or principal in principals:
            raise ValueError('Principals must be nonempty and unique')
        if role not in {'observer', 'lease-user', 'operator', 'catalog-admin'}:
            raise ValueError('Unknown identity role')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest) or digest in hashes:
            raise ValueError('Tokens require unique lowercase SHA256 digests')
        endpoints = row.get('allowed_endpoints')
        if endpoints is not None and (not isinstance(endpoints, list) or any(not isinstance(ep, str) or not ep for ep in endpoints)):
            raise ValueError('allowed_endpoints must list approved endpoint aliases')
        maximum = row.get('max_active_leases', 10 if role == 'lease-user' else None)
        ttl = row.get('max_ttl_seconds', 86400 if role == 'lease-user' else None)
        if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1):
            raise ValueError('max_active_leases must be positive')
        if ttl is not None and (isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not 0 < ttl <= 365 * 86400):
            raise ValueError('max_ttl_seconds must be positive and at most one year')
        principals.add(principal)
        hashes.add(digest)
        result.append((digest, Identity(principal, role, tuple(endpoints) if endpoints is not None else None, maximum, ttl)))
    return result


def authenticate(token: str, registry: list[tuple[str, Identity]]) -> Identity | None:
    digest = hashlib.sha256(token.encode()).hexdigest()
    result = None
    for candidate, identity in registry:
        if hmac.compare_digest(candidate, digest):
            result = identity
    return result
