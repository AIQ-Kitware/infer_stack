"""Tests for `infer-stack config …` settings + how they're honored."""

from __future__ import annotations


def test_config_init_yes_is_noninteractive(tmp_path, monkeypatch):
    monkeypatch.setenv('INFER_STACK_CONFIG_DIR', str(tmp_path))
    from infer_stack.cli.commands_meta import ConfigInitCLI
    from infer_stack.paths import load_settings, set_data_root

    set_data_root(None)
    # --yes must not prompt (no tty in tests) and must persist presets
    ConfigInitCLI.main(argv=['--yes', '--data-dir', str(tmp_path / 's'),
                             '--backend', 'compose'])
    settings = load_settings()
    assert settings['backend'] == 'compose'
    assert settings['data_dir'] == str(tmp_path / 's')


def test_config_init_announces_new_then_editing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFER_STACK_CONFIG_DIR', str(tmp_path))
    from infer_stack.cli.commands_meta import ConfigInitCLI
    from infer_stack.paths import set_data_root

    set_data_root(None)
    ConfigInitCLI.main(argv=['--yes', '--backend', 'compose'])
    assert 'from scratch' in capsys.readouterr().out          # first run = new
    ConfigInitCLI.main(argv=['--yes', '--backend', 'compose'])
    out = capsys.readouterr().out                              # second = editing
    assert 'editing the existing config' in out
    assert 'config edit' in out


def test_config_init_persists_all_known_settings(tmp_path, monkeypatch):
    monkeypatch.setenv('INFER_STACK_CONFIG_DIR', str(tmp_path))
    from infer_stack.cli.commands_meta import KNOWN_SETTINGS, ConfigInitCLI
    from infer_stack.paths import load_settings, set_data_root

    set_data_root(None)
    ConfigInitCLI.main(argv=['--yes', '--data-dir', str(tmp_path / 's'),
                             '--backend', 'compose'])
    settings = load_settings()
    # init now writes every known setting (incl. ui + skip_display_gpus), so
    # nothing the system honors is silently missing from a fresh config.
    assert set(settings) == set(KNOWN_SETTINGS)
    assert settings['skip_display_gpus'] is False     # default
    assert settings['ui'] is True                     # default


def test_config_init_preserves_other_settings(tmp_path, monkeypatch):
    monkeypatch.setenv('INFER_STACK_CONFIG_DIR', str(tmp_path))
    from infer_stack.cli.commands_meta import ConfigInitCLI, ConfigSetCLI
    from infer_stack.paths import load_settings, set_data_root

    set_data_root(None)
    ConfigInitCLI.main(argv=['--yes', '--backend', 'compose'])
    ConfigSetCLI.main(argv=['ui', 'false'])                    # an unrelated key
    ConfigInitCLI.main(argv=['--yes', '--backend', 'null'])    # re-init in place
    settings = load_settings()
    assert settings['ui'] is False                            # preserved
    assert settings['backend'] == 'null'


def test_config_init_fresh_resets_to_defaults(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('INFER_STACK_CONFIG_DIR', str(tmp_path))
    from infer_stack.cli.commands_meta import ConfigInitCLI, ConfigSetCLI
    from infer_stack.paths import load_settings, set_data_root

    set_data_root(None)
    ConfigInitCLI.main(argv=['--yes', '--backend', 'compose'])
    ConfigSetCLI.main(argv=['ui', 'false'])                   # customize ui
    ConfigInitCLI.main(argv=['--yes', '--fresh', '--backend', 'compose'])
    assert 'starting fresh' in capsys.readouterr().out
    settings = load_settings()
    # --fresh discards the customization and resets to the default (ui defaults on)
    assert settings['ui'] is True
    assert settings['backend'] == 'compose'


def test_config_set_get_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv('INFER_STACK_CONFIG_DIR', str(tmp_path))
    from infer_stack.cli.commands_meta import ConfigGetCLI, ConfigSetCLI
    from infer_stack.paths import load_settings

    ConfigSetCLI.main(argv=['backend', 'compose'])
    assert load_settings()['backend'] == 'compose'
    # get prints it (smoke; value asserted via load_settings)
    ConfigGetCLI.main(argv=['backend'])


def test_data_dir_setting_honored_by_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv('INFER_STACK_CONFIG_DIR', str(tmp_path))
    monkeypatch.delenv('INFER_STACK_DATA_DIR', raising=False)
    from infer_stack.cli.commands_meta import ConfigSetCLI
    from infer_stack.paths import data_root, set_data_root

    set_data_root(None)  # clear any process override
    ConfigSetCLI.main(argv=['data_dir', str(tmp_path / 'state')])
    assert data_root() == tmp_path / 'state'


def test_env_data_dir_beats_setting(tmp_path, monkeypatch):
    monkeypatch.setenv('INFER_STACK_CONFIG_DIR', str(tmp_path))
    from infer_stack.cli.commands_meta import ConfigSetCLI
    from infer_stack.paths import data_root, set_data_root

    set_data_root(None)
    ConfigSetCLI.main(argv=['data_dir', str(tmp_path / 'from-setting')])
    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path / 'from-env'))
    assert data_root() == tmp_path / 'from-env'   # env wins over setting


def test_backend_setting_resolved_by_make_backend(tmp_path, monkeypatch):
    monkeypatch.setenv('INFER_STACK_CONFIG_DIR', str(tmp_path))
    import infer_stack.cli.commands_leasing as mod
    import infer_stack.hardware as hw
    from infer_stack.cli.commands_leasing import AcquireCLI
    from infer_stack.cli.commands_meta import ConfigSetCLI

    seen = {}

    class FakeCompose:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(mod, 'ComposeBackend', FakeCompose)
    monkeypatch.setattr(hw, 'detect_inventory', lambda: {})

    # No persisted backend -> null (dry-run)
    cfg = AcquireCLI.cli(argv=['e'], strict=False)
    assert type(mod._make_backend(cfg)).__name__ == 'NullBackend'

    # Persisted backend -> compose, even without --backend
    ConfigSetCLI.main(argv=['backend', 'compose'])
    cfg = AcquireCLI.cli(argv=['e'], strict=False)
    assert isinstance(mod._make_backend(cfg), FakeCompose)

    # Explicit --backend still wins (null overrides the setting)
    cfg = AcquireCLI.cli(argv=['e', '--backend', 'null'], strict=False)
    assert type(mod._make_backend(cfg)).__name__ == 'NullBackend'
