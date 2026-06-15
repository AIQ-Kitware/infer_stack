from __future__ import annotations

from ..catalog import PROFILE_NAME_ALIASES
from ..catalog import profile_summary
from ..config import dump_yaml
from ..config import initial_config
from ..config import load_kubeai_resource_profiles
from ..config import load_yaml
from ..config import normalized_catalogs
from ..config import save_kubeai_resource_profiles
from ..config import save_yaml
from ..contracts import load_profile_contract
from ..diff_prompt import confirm_and_write
from ..kubeai_ops import deploy_rendered_artifacts
from ..renderer import render_from_lock
from ..verification import verify_profile
from pathlib import Path
from typing import Any
import json
import scriptconfig as scfg

from .context import (
    _apply_path_overrides,
    _arg_or_env,
    _as_mapping,
    apply_config_overrides,
    backend_name,
    build_plan,
    config_for_runtime,
    config_path,
    effective_allow_unsupported,
    effective_inventory,
    ensure_renderable,
    generated_dir,
    kubeai_generated_dir,
    load_config,
    models_path,
    plan_path,
    runtime_dir_for_config,
    save_plan,
)
from .compose import _compose_up_with_router_recreate
from .options import (
    _AllowUnsupportedMixin,
    _BackendOverrideMixin,
    _ClusterOverridesMixin,
    _ComposeOverrideMixin,
    _PathOverridesMixin,
    _PlanOverridesCLI,
    _PortOverridesMixin,
    _ProfileOverrideMixin,
    _SimulateHardwareMixin,
    _SwitchPathOverridesCLI,
)

# ---------------------------------------------------------------------------
# Profile / config management commands
# ---------------------------------------------------------------------------


class InitCLI(_PathOverridesMixin):
    """Write a fresh config.yaml + empty models.yaml under config_root()."""

    force = scfg.Value(
        False, isflag=True, help='Overwrite an existing config.yaml.'
    )
    yes = scfg.Value(
        False,
        isflag=True,
        short_alias=['y'],
        help='Apply the writes without prompting. Without this, init shows a '
        'per-file diff and asks for confirmation.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg_path = config_path()
        if cfg_path.exists() and not config.force:
            raise SystemExit(
                'config.yaml already exists. Use --force to overwrite.'
            )
        planned = {cfg_path: dump_yaml(initial_config())}
        if not models_path().exists():
            planned[models_path()] = dump_yaml({'models': {}, 'profiles': {}})
        if not confirm_and_write(
            planned, assume_yes=bool(config.yes), title='Pending init'
        ):
            raise SystemExit('Aborted by user; no files were written.')
        print(f'Wrote {cfg_path}')
        return 0


class SetupCLI(
    _PathOverridesMixin,
    _ProfileOverrideMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
):
    """Create / update config.yaml with the requested overrides applied."""

    reset = scfg.Value(
        False,
        isflag=True,
        help='Start from default config values before applying overrides.',
    )
    resource_profiles_file = scfg.Value(
        None,
        type=str,
        help='For kubeai setups, sync a local Helm values file with resourceProfiles into kubeai-values.local.yaml.',
    )
    yes = scfg.Value(
        False,
        isflag=True,
        short_alias=['y'],
        help='Apply the config writes without prompting. Without this, setup '
        'shows a per-file diff and asks for confirmation.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg_path = config_path()
        if cfg_path.exists() and not config.reset:
            cfg = load_yaml(cfg_path)
        else:
            cfg = initial_config()
        cfg = apply_config_overrides(cfg, config)
        planned = {cfg_path: dump_yaml(cfg)}
        if not models_path().exists():
            planned[models_path()] = dump_yaml({'models': {}, 'profiles': {}})
        if not confirm_and_write(
            planned, assume_yes=bool(config.yes), title='Pending setup'
        ):
            raise SystemExit('Aborted by user; no files were written.')
        if config.resource_profiles_file:
            # User-supplied path: anchor on CWD so a typed relative path
            # behaves as the user expects.
            source = Path(config.resource_profiles_file)
            if not source.is_absolute():
                source = Path.cwd() / source
            values_doc = load_yaml(source)
            if 'resourceProfiles' not in values_doc:
                raise SystemExit(
                    f'{source} is missing a top-level resourceProfiles map'
                )
            target = save_kubeai_resource_profiles(values_doc)
            if plan_path(cfg).exists():
                plan_path(cfg).unlink()
            print(f'Wrote {target}')
        print(f'Wrote {cfg_path}')
        print(
            f'Configured backend={cfg.get("backend", "compose")} '
            f'active_profile={cfg.get("active_profile", "") or "<unset>"}'
        )
        return 0


class ResolveCLI(_PlanOverridesCLI):
    """Resolve the active profile into a deployment dict and print it."""

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        save_plan(plan, cfg)
        print(json.dumps(plan['deployment'], indent=2))
        return 0


class ValidateCLI(_PlanOverridesCLI):
    """Resolve + validate the active profile; exit non-zero on validation errors."""

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        save_plan(plan, cfg)
        print(json.dumps(plan['validated'], indent=2))
        return 0 if plan['validated']['ok'] else 2


class LockCLI(_PlanOverridesCLI):
    """Write plan.yaml after resolving + validating the active profile."""

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        if not plan['validated']['ok'] and not plan['allow_unsupported']:
            raise SystemExit(
                'Refusing to write plan.yaml because validation failed. Use --allow-unsupported to override.'
            )
        save_plan(plan, cfg)
        print(json.dumps(plan, indent=2))
        return 0


class RenderCLI(_PlanOverridesCLI):
    """Render the active profile's deployment into compose/kubeai artifacts."""

    yes = scfg.Value(
        False,
        isflag=True,
        short_alias=['y'],
        help='Apply rendered changes without prompting. Without this, render shows a per-file diff and asks for confirmation.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        ensure_renderable(plan)
        save_plan(plan, cfg)
        render_from_lock(plan, assume_yes=bool(config.yes))
        print(f'Wrote {plan_path(cfg)}')
        if backend_name(cfg) == 'kubeai':
            print(f'Rendered KubeAI artifacts into {kubeai_generated_dir(cfg)}')
        else:
            print(f'Rendered Compose into {generated_dir(cfg)}')
            print(
                f'Rendered mounted runtime files into {runtime_dir_for_config(cfg)}'
            )
        return 0


class SwitchCLI(_SwitchPathOverridesCLI):
    """Persist a new active_profile and re-render (optionally re-applying)."""

    profile = scfg.Value(
        None, type=str, position=1, help='Profile name to switch to.'
    )
    apply = scfg.Value(
        False,
        isflag=True,
        help='Also apply the new profile to the running stack.',
    )
    yes = scfg.Value(
        False,
        isflag=True,
        short_alias=['y'],
        help='Apply rendered changes without prompting.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if not config.profile:
            raise SystemExit('switch: missing required profile name')
        persisted_cfg = load_config()
        persisted_cfg['active_profile'] = config.profile
        if not confirm_and_write(
            {config_path(): dump_yaml(persisted_cfg)},
            assume_yes=bool(config.yes),
            title='Pending active_profile change',
        ):
            raise SystemExit('Aborted by user; no files were written.')
        cfg = apply_config_overrides(persisted_cfg, config)

        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        ensure_renderable(plan)
        save_plan(plan, cfg)
        render_from_lock(plan, assume_yes=bool(config.yes))
        if config.apply:
            if backend_name(cfg) == 'compose':
                _compose_up_with_router_recreate(cfg, detach=True)
            else:
                deploy_rendered_artifacts(plan['deployment'])
        print(f'Switched active_profile to {config.profile}')
        return 0


class ListModelsCLI(_PathOverridesMixin):
    """Print every model in the merged catalog."""

    __command__ = 'list-models'

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = load_config() if config_path().exists() else initial_config()
        cats = normalized_catalogs(cfg)
        for name, model in cats.get('models', {}).items():
            ref = model.get('hf_model_id') or model.get('url', '')
            print(f'{name}: {ref}')
        return 0


class ListProfilesCLI(_PathOverridesMixin):
    """Print every serving profile in the merged catalog."""

    __command__ = 'list-profiles'

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = load_config() if config_path().exists() else initial_config()
        cats = normalized_catalogs(cfg)
        profiles = cats.get('profiles', {})
        hidden_legacy = set(PROFILE_NAME_ALIASES)
        for name, profile in profiles.items():
            if name in hidden_legacy:
                continue
            if profile.get('kind') == 'invalid-profile':
                continue
            summary = profile_summary(profile)
            providers = ','.join(summary.get('providers', [])) or 'none'
            print(
                f'{name}: providers={providers} gateway={summary["gateway"]} '
                f'frontend={summary["frontend"]} frontend_provider={summary["frontend_provider"]} '
                f'routes={summary["route_count"]}'
            )
        return 0


class ExplainCLI(_PathOverridesMixin):
    """Pretty-print a YAML file (defaults to the current plan.yaml)."""

    __command__ = 'explain'

    file = scfg.Value(
        None, type=str, help='Path to read (default: current plan.yaml).'
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if config.file:
            target = Path(config.file)
            if not target.is_absolute():
                target = Path.cwd() / target
        else:
            target = plan_path()
        if not target.exists():
            raise SystemExit(f'Missing file: {target}')
        print(json.dumps(load_yaml(target), indent=2))
        return 0


class DescribeProfileCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _AllowUnsupportedMixin,
    _SimulateHardwareMixin,
):
    """Print the profile contract for a given profile name."""

    __command__ = 'describe-profile'

    profile = scfg.Value(
        None, type=str, position=1, help='Profile name to describe.'
    )
    format = scfg.Value('yaml', choices=['json', 'yaml'])
    output = scfg.Value(
        None, type=str, help='Write to this file instead of stdout.'
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        overrides = _as_mapping(config)
        if not overrides.get('profile'):
            raise SystemExit('describe-profile: missing required profile name')
        contract = load_profile_contract(
            overrides['profile'],
            backend=_arg_or_env(overrides, 'backend', 'INFER_STACK_BACKEND'),
            simulate_hardware_spec=overrides.get('simulate_hardware'),
        )
        return _print_structured(
            contract, overrides['format'], overrides.get('output')
        )


def _print_structured(
    data: dict[str, Any], fmt: str, output: str | None
) -> int:
    if fmt == 'yaml':
        import yaml

        text = yaml.safe_dump(data, sort_keys=False)
    else:
        text = json.dumps(data, indent=2)
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            text + ('' if text.endswith('\n') else '\n'), encoding='utf-8'
        )
        print(f'Wrote {target}')
        return 0
    print(text)
    return 0


class VerifyProfileCLI(_SwitchPathOverridesCLI):
    """Sanity-check a resolved profile (post-render expectations)."""

    __command__ = 'verify-profile'

    profile = scfg.Value(
        None, type=str, position=1, help='Profile name to verify.'
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if not config.profile:
            raise SystemExit('verify-profile: missing required profile name')
        cfg = config_for_runtime(config)
        plan = build_plan(
            cfg,
            profile_name=config.profile,
            allow_unsupported=effective_allow_unsupported(config, cfg),
            inventory=effective_inventory(config),
        )
        result = verify_profile(plan['deployment'])
        print(json.dumps(result, indent=2))
        return 0 if result['ok'] else 2


class KubeaiSyncResourceProfilesCLI(_PathOverridesMixin):
    """Pull a Helm ``resourceProfiles`` values file into kubeai-values.local.yaml."""

    __command__ = 'kubeai-sync-resource-profiles'

    from_file = scfg.Value(
        None,
        type=str,
        required=True,
        help='Helm values file with a top-level resourceProfiles map.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config, allow_missing=True)
        # User-supplied path: anchor on CWD.
        source = Path(config.from_file)
        if not source.is_absolute():
            source = Path.cwd() / source
        values_doc = load_yaml(source)
        if 'resourceProfiles' not in values_doc:
            raise SystemExit(
                f'{source} is missing a top-level resourceProfiles map'
            )
        target = save_kubeai_resource_profiles(values_doc)
        if plan_path(cfg).exists():
            plan_path(cfg).unlink()
        profiles, _, _ = load_kubeai_resource_profiles()
        print(f'Wrote {target}')
        print(f'Synced {len(profiles)} KubeAI resource profile(s)')
        return 0
