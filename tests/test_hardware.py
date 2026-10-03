"""Focused tests for hardware inventory discovery."""

from __future__ import annotations

from infer_stack import hardware


def test_detect_inventory_adds_compute_capability(monkeypatch):
    def fake_run(cmd, **kwargs):
        query = next((arg for arg in cmd if arg.startswith('--query-gpu=')), '')
        if query == '--query-gpu=index,uuid,name,memory.total,display_active':
            return (
                '0, GPU-aaa, NVIDIA RTX PRO 6000 Blackwell, 97887, Disabled\n'
                '3, GPU-bbb, Quadro RTX 8000, 49152, Enabled\n'
            )
        if query == '--query-gpu=index,compute_cap':
            return '0, 12.0\n3, 7.5\n'
        raise AssertionError(cmd)

    monkeypatch.setattr(hardware, '_run', fake_run)
    inv = hardware.detect_inventory()
    assert inv['gpu_count'] == 2
    assert [gpu['index'] for gpu in inv['gpus']] == [0, 3]
    assert [gpu['compute_cap'] for gpu in inv['gpus']] == [12.0, 7.5]
    assert inv['gpus'][0]['memory_gib'] == 95.59
    assert inv['gpus'][1]['display_active'] is True


def test_detect_inventory_tolerates_missing_compute_capability(monkeypatch):
    def fake_run(cmd, **kwargs):
        query = next((arg for arg in cmd if arg.startswith('--query-gpu=')), '')
        if query == '--query-gpu=index,uuid,name,memory.total,display_active':
            return '0, GPU-aaa, NVIDIA A40, 46068, Disabled\n'
        if query == '--query-gpu=index,compute_cap':
            return ''
        raise AssertionError(cmd)

    monkeypatch.setattr(hardware, '_run', fake_run)
    inv = hardware.detect_inventory()
    assert inv['gpu_count'] == 1
    assert inv['gpus'][0]['name'] == 'NVIDIA A40'
    assert 'compute_cap' not in inv['gpus'][0]
