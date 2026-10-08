"""The same lease/operation contract on SQLite and a real optional PostgreSQL.

Set INFER_STACK_TEST_POSTGRES_DSN to a disposable database. Each case creates
and drops only its own random schema; no production/public tables are touched.
"""
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from infer_stack.control.operations import Operations
from infer_stack.leasing import Catalog, Ledger, SqliteStore


@pytest.fixture(params=['sqlite', 'postgres'])
def stores(request, tmp_path):
    opened = []
    if request.param == 'sqlite':
        def create():
            store = SqliteStore(tmp_path / 'ledger.db')
            opened.append(store)
            return store
        def cleanup():
            pass
    else:
        dsn = os.environ.get('INFER_STACK_TEST_POSTGRES_DSN')
        if not dsn:
            pytest.skip('Set INFER_STACK_TEST_POSTGRES_DSN for real PostgreSQL contract tests')
        psycopg = pytest.importorskip('psycopg')
        from psycopg import sql
        from psycopg.conninfo import make_conninfo

        from infer_stack.leasing.postgres import PostgresStore
        schema = 'infer_stack_test_' + uuid.uuid4().hex
        admin = psycopg.connect(dsn, autocommit=True)
        admin.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        scoped = make_conninfo(dsn, options=f'-c search_path={schema}')
        def create():
            store = PostgresStore(scoped)
            opened.append(store)
            return store
        def cleanup():
            admin.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
            admin.close()
    try:
        yield create
    finally:
        for store in opened:
            store.close()
        cleanup()


def requests():
    return Catalog.from_dict({'models': {'m': {'source': 'hf://org/m'}},
                              'endpoints': {'q': {'engine': 'vllm', 'model': 'm'}}}).resolve_requests(['q'])


def test_concurrent_coalescing_and_distinct_leases(stores):
    authorities = [Ledger(stores()) for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        acquired = list(executor.map(lambda ledger: ledger.acquire('alice', requests()), authorities))
    assert len({a.lease.id for a in acquired}) == 8
    assert len({a.deployments[0].id for a in acquired}) == 1
    ledger = authorities[0]
    assert ledger.get_deployment(acquired[0].deployments[0].id).demand == 8
    for outcome in acquired:
        ledger.release(outcome.lease.id)
    assert ledger.get_deployment(acquired[0].deployments[0].id).state == 'idle'


def test_rollback_on_interrupt_and_durable_publication_marker(stores):
    store = stores()
    with pytest.raises(KeyboardInterrupt):
        with store.transaction() as connection:
            connection.execute('INSERT INTO meta(key,value) VALUES (?,?)', ('interrupted', 'value'))
            raise KeyboardInterrupt
    assert store.meta_json('interrupted') is None
    store.mark_publication_pending(apply_requested=True, interrupted=False)
    first = store.publication_pending()
    store.mark_publication_pending(apply_requested=True, interrupted=True)
    assert not store.clear_publication_pending(first['version'])
    reopened = stores()
    latest = reopened.publication_pending()
    assert latest['interrupted']
    assert reopened.clear_publication_pending(latest['version'])
    assert store.publication_pending() is None


def test_operations_share_authority_and_survive_reopen(stores):
    operations = [Operations(stores()) for _ in range(8)]
    request = {'action': 'acquire', 'client': {'endpoints': ['q']}}
    with ThreadPoolExecutor(max_workers=8) as executor:
        accepted = list(executor.map(lambda op: op.submit('alice', 'same-key', request), operations))
    assert len({op['id'] for op in accepted}) == 1
    operations[0].transition(accepted[0]['id'], 'succeeded', result={'lease_id': 'lease-1'})
    assert Operations(stores()).find('alice', 'same-key')['result'] == {'lease_id': 'lease-1'}
    audited = Operations(stores()).audit(limit=1)
    assert audited[0]['state'] == 'queued'
    assert Operations(stores()).audit(after=audited[0]['id'])[0]['state'] == 'succeeded'


def test_server_authority_excludes_a_second_process_connection(stores):
    first, second = stores(), stores()
    with first.service_lock():
        with pytest.raises(ValueError, match='Another control server'):
            with second.service_lock():
                pytest.fail('two services became authoritative')
    with second.service_lock():
        pass


def test_fractional_ttl_and_generations(stores):
    now = [1791412351.123456]
    store = stores()
    ledger = Ledger(store, clock=lambda: now[0])
    lease = ledger.acquire('alice', requests(), ttl_seconds=.5).lease
    assert ledger.get_lease(lease.id).expires_at == now[0] + .5
    generation = store.desired_generation()
    store.set_applied_generation(generation)
    store.set_applied_generation(generation - 1)
    assert store.applied_generation() == generation
    now[0] += .6
    assert ledger.sweep().expired_lease_ids == [lease.id]


def test_incompatible_schema_is_refused_on_reopen(stores):
    store = stores()
    store._conn.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    with pytest.raises(ValueError, match='schema version'):
        stores()


def test_concurrent_pending_lease_quota_is_transactional(stores):
    from infer_stack.control.auth import Identity
    from infer_stack.control.service import ControlService
    from infer_stack.leasing import Controller
    from infer_stack.leasing.backend import NullBackend

    catalog = Catalog.from_dict({'models': {'m': {'source': 'hf://org/m'}},
                                 'endpoints': {'q': {'engine': 'vllm', 'model': 'm'}}})
    services = [ControlService(Controller(Ledger(stores()), NullBackend()), catalog) for _ in range(8)]
    identity = Identity('alice', 'lease-user', max_active_leases=1, max_ttl_seconds=60)
    def acquire(index):
        try:
            submit = services[index].acquire if index % 2 else services[index].access
            submit('alice', f'quota-{index}', {'endpoints': ['q'], 'ttl_seconds': 60}, identity=identity)
            return True
        except PermissionError:
            return False
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sum(executor.map(acquire, range(8))) == 1
    assert len(services[0].operations.all()) == 1


def test_lost_postgres_service_session_refuses_publication(stores):
    store = stores()
    if store.path is not None:
        pytest.skip('PostgreSQL session-authority failure contract')
    import psycopg

    with store.service_lock():
        store._service_connection.close()
        with pytest.raises(psycopg.OperationalError):
            with store.publication_lock():
                pytest.fail('Lost service authority reached backend publication')
