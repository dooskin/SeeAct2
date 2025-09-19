"""Persona Prompt Pack + GA-powered Persona Pool (decoupled).

Subpackages:
- adapter: Neon/GA aggregation and table management
- builder: cohort normalization, intent, k-anon, sampling, pool generation
- prompts: UXAgent-aligned templates + generator + vendor exemplars
- scrape: Shopify vocabulary scraper
"""

from .builder.pool_builder import (
    build_master_pool,
    save_pool_artifacts,
)
from .prompts.generator import (
    render_shop_browse_prompt,
    write_prompt_module,
)

__all__ = [
    "build_master_pool",
    "save_pool_artifacts",
    "render_shop_browse_prompt",
    "write_prompt_module",
]

