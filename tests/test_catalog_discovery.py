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
    assert not any(r['name'].startswith(MAYA) for r in profile_choices(
        simulate_inventory('2x24@8.9')))
    assert not any(r['managed_build'] for r in profile_choices(
        simulate_inventory('2x24@8.6'), kubeai=True))


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


def test_incompatible_profile_cannot_be_imported():
    with pytest.raises(ValueError, match='does not fit'):
        endpoint_fragment(discovery_catalog(simulate_inventory('1x16@8.6')), MAYA)
