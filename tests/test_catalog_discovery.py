"""Exact registry imports share the suggestion compiler without seeding siblings."""
from __future__ import annotations

import copy

import pytest

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing.catalog_edit import merge_catalog_fragment
from infer_stack.leasing.discovery import (
    discovery_catalog, endpoint_fragment, profile_choices, registry_endpoint_names,
)
from infer_stack.runtime_images import MAYA_IMAGE


MAYA = 'inkling-2-mini-maya'
GEMMAS = {'gemma4-e4b', 'gemma4-26b', 'gemma4-31b'}


def test_registry_preserves_optional_models_and_variants():
    inv = simulate_inventory('2x24@8.6')
    default = discovery_catalog(inv, all_models=False)['endpoints']
    all_eps = discovery_catalog(inv)['endpoints']
    assert {MAYA, MAYA + '-long', 'gemma4-e2b'} <= default.keys()
    assert not (GEMMAS | {MAYA + '-huge'}) & default.keys()
    assert GEMMAS | {MAYA + '-huge'} <= all_eps.keys()
    assert registry_endpoint_names() >= all_eps.keys()
    assert all_eps[MAYA + '-long']['runtime']['image'] == MAYA_IMAGE
    assert all_eps[MAYA + '-long']['runtime']['max_model_len'] == 262144
    assert all_eps[MAYA + '-huge']['runtime']['env']['MAYA_DSPARK'] == '0'


def test_choices_expose_real_variants_and_image_recipe():
    choices = profile_choices(simulate_inventory('2x24@8.6'))
    rows = {r['name']: r for r in choices}
    assert rows[MAYA + '-long']['tier'] == 'frontier'
    assert rows[MAYA + '-long']['gpu_count'] == 1
    assert rows[MAYA + '-long']['managed_build']
    assert rows[MAYA + '-huge']['tier'] == 'optional'
    assert rows['gemma4-31b']['tier'] == 'optional'
    assert set(rows[MAYA + '-long']['fragment']['models']) == {MAYA}
    assert set(rows[MAYA + '-long']['fragment']['endpoints']) == {MAYA + '-long'}
    wrong_arch = {r['name']: r for r in profile_choices(simulate_inventory('2x24@8.9'))}
    assert not wrong_arch[MAYA]['hardware_fit']
    assert 'CUDA compute 8.6' in wrong_arch[MAYA]['warnings'][0]
    assert wrong_arch[MAYA]['managed_build']
    cluster = {r['name']: r for r in profile_choices(
        simulate_inventory('2x24@8.6'), kubeai=True)}
    assert cluster[MAYA]['hardware_fit'] and not cluster[MAYA]['backend_compatible']
    assert 'KubeAI' in cluster[MAYA]['warnings'][0]
    assert cluster[MAYA]['managed_build']


def test_exact_import_preserves_existing_and_is_idempotent():
    fragment = endpoint_fragment(discovery_catalog(simulate_inventory('2x24@8.6')),
                                 MAYA + '-long')
    existing = {'models': {'mine': {'source': 'hf://me/model'}},
                'endpoints': {'mine': {'engine': 'vllm', 'model': 'mine'}}}
    original = copy.deepcopy(existing)
    updated, added = merge_catalog_fragment(existing, fragment)
    assert existing == original
    assert set(updated['endpoints']) == {'mine', MAYA + '-long'}
    assert set(added) == {f'models:{MAYA}', f'endpoints:{MAYA}-long'}
    assert merge_catalog_fragment(updated, fragment) == (updated, [])


@pytest.mark.parametrize('section', ['models', 'endpoints'])
def test_conflict_refuses_before_partial_import(section):
    fragment = endpoint_fragment(discovery_catalog(simulate_inventory('2x24@8.6')),
                                 MAYA + '-long')
    existing = copy.deepcopy(fragment)
    name = next(iter(existing[section]))
    existing[section][name]['revision' if section == 'models' else 'served_name'] = 'mine'
    original = copy.deepcopy(existing)
    with pytest.raises(ValueError, match='already exists and differs'):
        merge_catalog_fragment(existing, fragment)
    assert existing == original


def test_automatic_suggestions_still_filter_incompatible_profiles():
    with pytest.raises(ValueError, match='does not fit'):
        endpoint_fragment(discovery_catalog(simulate_inventory('1x16@8.6')), MAYA)


@pytest.mark.parametrize('inventory', [
    {'gpus': []}, simulate_inventory('2x24@8.6'),
    simulate_inventory('4x96@12.0'), simulate_inventory('192@9.0,4x96@12.0'),
])
def test_complete_registry_and_fitting_recipe_identity(inventory):
    from infer_stack.leasing import Catalog

    fitting = discovery_catalog(inventory)['endpoints']
    rows = {r['name']: r for r in profile_choices(inventory)}
    assert set(rows) == registry_endpoint_names()
    for name, row in rows.items():
        Catalog.from_dict(row['fragment'])
        ep = row['fragment']['endpoints'][name]
        assert row['hardware_fit'] == (name in fitting)
        if name in fitting:
            assert ep == fitting[name]
        else:
            assert 'gpu_indices' not in ep.get('placement', {})
            assert row['warnings']


def test_deepseek_profiles_are_visible_with_advisory_constraints():
    name = 'deepseek-v4-flash-0731'
    variant = name + '-tp2-experimental'
    inv = simulate_inventory('2x24@8.6')
    rows = {r['name']: r for r in profile_choices(inv)}
    assert not rows[name]['hardware_fit']
    assert rows[name]['gpu_count'] == 4
    assert rows[name]['min_vram_gib'] == 80
    assert 'CUDA compute 12.0' in rows[name]['warnings'][0]
    assert 'passed generation readiness' in rows[name]['note']
    assert not rows[variant]['hardware_fit']
    assert rows[variant]['gpu_count'] == 2
    assert rows[variant]['context'] == 32768
    assert rows[variant]['tier'] == 'optional'
    assert 'unverified' in rows[variant]['note']
    # Exact import does not relax floors or fabricate hardware pins.
    fragment = rows[name]['fragment']
    updated, added = merge_catalog_fragment({}, fragment)
    assert set(updated['endpoints']) == {name}
    assert updated['endpoints'][name]['placement'] == {'min_vram_gib': 80}
    assert merge_catalog_fragment(updated, fragment) == (updated, [])
    # A two-GPU host only gets the experiment through explicit discovery,
    # not through default automatic suggestions or an orphan model entry.
    big = simulate_inventory('2x96@12.0')
    normal = discovery_catalog(big, all_models=False)
    assert name not in normal['models'] and variant not in normal['endpoints']
    assert variant in discovery_catalog(big)['endpoints']
