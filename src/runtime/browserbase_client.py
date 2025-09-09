from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import requests


DEFAULT_API_BASE = "https://api.browserbase.com/v1"


class BrowserbaseError(RuntimeError):
    pass


def _pick_cdp_url(data: Dict[str, Any]) -> Optional[str]:
    for key in ("wsEndpoint", "cdpUrl", "wsUrl", "connectUrl", "url"):
        val = data.get(key)
        if isinstance(val, str) and val.startswith("ws"):
            return val
    session = data.get("session") or {}
    if isinstance(session, dict):
        for key in ("wsEndpoint", "cdpUrl", "wsUrl", "connectUrl", "url"):
            val = session.get(key)
            if isinstance(val, str) and val.startswith("ws"):
                return val
    return None


def _pick_session_id(data: Dict[str, Any]) -> Optional[str]:
    for key in ("id", "sessionId", "session_id"):
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
    session = data.get("session") or {}
    if isinstance(session, dict):
        for key in ("id", "sessionId", "session_id"):
            val = session.get(key)
            if isinstance(val, str) and val:
                return val
    return None


def resolve_credentials(project_id: Optional[str], api_key: Optional[str]) -> Tuple[str, str]:
    pid = project_id or os.getenv("BROWSERBASE_PROJECT_ID")
    key = api_key or os.getenv("BROWSERBASE_API_KEY")
    if not pid:
        raise BrowserbaseError("Missing Browserbase project id. Set BROWSERBASE_PROJECT_ID or provide in config.")
    if not key:
        raise BrowserbaseError("Missing Browserbase API key. Set BROWSERBASE_API_KEY or provide in config.")
    return pid, key


def create_session(project_id: str, api_key: str, *, api_base: Optional[str] = None, timeout_sec: int = 30) -> Tuple[str, Optional[str]]:
    base = api_base or os.getenv("BROWSERBASE_API_BASE", DEFAULT_API_BASE)
    url = f"{base.rstrip('/')}/sessions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload: Dict[str, Any] = {"projectId": project_id}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    if resp.status_code // 100 != 2:
        raise BrowserbaseError(f"Failed to create Browserbase session: {resp.status_code} {resp.text}")
    data = resp.json() if resp.content else {}
    cdp_url = _pick_cdp_url(data)
    session_id = _pick_session_id(data)
    if not cdp_url:
        raise BrowserbaseError(f"Browserbase session created but no CDP URL returned: {data}")
    return cdp_url, session_id


def close_session(session_id: str, api_key: str, *, api_base: Optional[str] = None, timeout_sec: int = 10) -> None:
    base = api_base or os.getenv("BROWSERBASE_API_BASE", DEFAULT_API_BASE)
    url = f"{base.rstrip('/')}/sessions/{session_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    try:
        requests.delete(url, headers=headers, timeout=timeout_sec)
    except Exception:
        pass
