from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import toml as tomllib  # type: ignore


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
PROFILE_DIR = CONFIG_DIR / "profiles"
DEFAULT_BASE_CONFIG = CONFIG_DIR / "base.toml"
DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parents[2] / "site_manifest"
MANIFEST_ENV_VAR = "SEEACT_MANIFEST_DIR"


class SettingsLoadError(RuntimeError):
    pass


def _load_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SettingsLoadError(f"Config file not found: {path}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)  # type: ignore[arg-type]
    if not isinstance(data, dict):
        raise SettingsLoadError(f"Config file {path} did not produce a dictionary")
    return data


def _merge_dicts(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand_env(data: Any) -> Any:
    if isinstance(data, str):
        return os.path.expandvars(data)
    if isinstance(data, dict):
        return {k: _expand_env(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_env(v) for v in data]
    return data


_PATH_FIELDS: Tuple[Tuple[str, ...], ...] = (
    ("basic", "save_file_dir"),
    ("experiment", "task_file_path"),
    ("runner", "metrics_dir"),
    ("manifest", "cache_dir"),
    ("manifest", "dir"),
)


def _resolve_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    for keys in _PATH_FIELDS:
        container = cfg
        for key in keys[:-1]:
            if not isinstance(container, dict):
                container = None
                break
            container = container.get(key)
        if not isinstance(container, dict):
            continue
        leaf = keys[-1]
        value = container.get(leaf)
        if isinstance(value, str) and value:
            path = Path(os.path.expanduser(value))
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            container[leaf] = str(path)
    return cfg


def load_settings(
    config_path: Path | None = None,
    profiles: Sequence[str] | None = None,
) -> Tuple[Dict[str, Any], Path]:
    """Load configuration as a merged dictionary and return with its base directory."""

    if config_path is not None:
        base_path = Path(config_path).resolve()
        base_dir = base_path.parent
        config = _load_toml(base_path)
    else:
        base_path = DEFAULT_BASE_CONFIG
        base_dir = base_path.parent
        config = _load_toml(base_path)

    for profile in profiles or []:
        profile_path = PROFILE_DIR / f"{profile}.toml"
        config = _merge_dicts(config, _load_toml(profile_path))

    config = _expand_env(config)
    config = _resolve_paths(config, base_dir)
    manifest_cfg = config.setdefault("manifest", {})
    manifest_dir_value = os.getenv(MANIFEST_ENV_VAR) or manifest_cfg.get("dir") or manifest_cfg.get("cache_dir")
    if manifest_dir_value:
        manifest_dir = Path(str(manifest_dir_value)).expanduser()
        if not manifest_dir.is_absolute():
            manifest_dir = (base_dir / manifest_dir).resolve()
    else:
        manifest_dir = DEFAULT_MANIFEST_DIR
    manifest_cfg["dir"] = str(manifest_dir)
    config.setdefault("__meta", {})["config_dir"] = str(base_dir)
    config["__meta"]["config_path"] = str(base_path)
    if profiles:
        config["__meta"]["profiles"] = list(profiles)
    return config, base_dir


__all__ = ["load_settings", "SettingsLoadError", "DEFAULT_BASE_CONFIG", "PROFILE_DIR"]
