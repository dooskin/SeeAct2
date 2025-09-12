#!/usr/bin/env python3
"""
Persona builder CLI (shim).

Delegates implementation to the package: seeact.personas.build_personas_yaml.
"""
from __future__ import annotations

from seeact.personas import build_personas_yaml  # re-export for tests
