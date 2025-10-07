"""
Manifest loading utilities for SeeAct.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Manifest:
    """Manifest data structure"""
    data: Dict[str, Any]
    domain: str
    cache_dir: Optional[Path] = None


def load_manifest(domain: str, cache_dir: Optional[Path] = None) -> Optional[Manifest]:
    """
    Load manifest for a domain.
    
    Args:
        domain: The domain to load manifest for
        cache_dir: Optional cache directory
        
    Returns:
        Manifest data or None if not found
    """
    # For now, return None (no manifest)
    # In a real implementation, this would load site-specific manifests
    return None
