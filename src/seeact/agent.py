# -*- coding: utf-8 -*-
# Copyright (c) 2024 OSU Natural Language Processing Group
#
# Licensed under the OpenRAIL-S License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.licenses.ai/ai-pubs-open-rails-vz1
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SeeAct Agent

Personas (decoupled):
- A separate module under `src/personas/` builds a GA-powered master pool (1000 personas), renders UXAgent-style prompts, and optionally scrapes Shopify vocab. It is intentionally decoupled from the agent and runner.
- Local (no DB) CLI:
    PYTHONPATH=src python -m personas.cli seed-demo --data-dir data/personas
    PYTHONPATH=src python -m personas.cli sample --size 10 --ids-out persona_ids.json --data-dir data/personas
    PYTHONPATH=src python -m personas.cli generate-prompts --site-domain yourstore.com --ids-file persona_ids.json --data-dir data/personas --out-dir data/personas/prompts
- DB-backed API build: `POST /v1/personas/generate-master` with `include_prompts=true` renders prompts for all 1000; charts data via `/v1/personas/traffic-summary` and `/v1/personas/behavior-match`.
- Runner usage: provide a personas YAML (id→weight map) via `--personas`; the runner tags each task with `persona_id`. This agent accepts any prompt string produced by the personas generator without code changes here.
"""

import json
import logging
import os
import random
import traceback
from datetime import datetime
from os.path import dirname
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import playwright

from seeact.Exceptions import TaskExecutionRetryError

try:
    import toml  # type: ignore
except Exception:
    class _TomlStub:  # type: ignore
        @staticmethod
        def load(fp):
            return {}
        @staticmethod
        def dump(obj, fp):
            try:
                fp.write("# TOML output unavailable (toml not installed)\n")
            except Exception:
                pass
    toml = _TomlStub()  # type: ignore
import importlib
import asyncio
import copy
import re
import time

from .data_utils.format_prompt_utils import get_index_from_option_name, generate_new_query_prompt, \
    generate_new_referring_prompt, format_options, generate_option_name
from .demo_utils.browser_helper import normal_launch_async, normal_new_context_async, \
    get_interactive_elements_with_playwright, select_option, saveconfig, auto_dismiss_overlays, register_overlay_hint
from .demo_utils.crawler_helper import get_random_link
from .demo_utils.format_prompt import format_choices, postprocess_action_lmm, postprocess_action_lmm_pixel
try:
    from .demo_utils.inference_engine import engine_factory  # type: ignore
except Exception:
    engine_factory = None  # type: ignore
# Optional: Browserbase CDP session helpers
try:
    from .runtime.browserbase_client import (
        resolve_credentials as bb_resolve,
        create_session as bb_create,
        close_session as bb_close,
    )
except Exception:
    bb_resolve = bb_create = bb_close = None  # type: ignore

from .utils.manifest_loader import load_manifest as load_manifest_from_dir, ManifestRecord


class SeeActAgent:
    def __init__(self,
                 config_path=None,
                 config=None,
                 save_file_dir="seeact_agent_files",
                 default_task='Find the pdf of the paper "GPT-4V(ision) is a Generalist Web Agent, if Grounded"',
                 default_website="https://www.google.com/",
                 input_info=["screenshot"],
                 grounding_strategy="text_choice_som",  # [...,'pixel_2_stage']
                 crawler_mode=False,
                 crawler_max_steps=10,
                 max_auto_op=50,
                 max_continuous_no_op=5,
                 highlight=False,
                 headless=False,
                 args=[],
                 browser_app="chrome",
                 persistant=False,
                 persistant_user_path="",
                 save_video=False,
                 viewport={
                     "width": 1280,
                     "height": 720
                 },
                 tracing=False,
                 trace={
                     "screenshots": True,
                     "snapshots": True,
                     "sources": True
                 },
                 rate_limit=-1,
                 model="gpt-4o",
                 temperature=0.9,
                 worker_id: Optional[int] = None,
                 ):

        try:
            if config is not None:
                config = copy.deepcopy(config)
                meta = config.pop("__meta", {}) if isinstance(config, dict) else {}
                if config_path is None:
                    config_path = meta.get("config_path")
                self._config_dir = Path(meta.get("config_dir", os.getcwd()))
            elif config_path is not None:
                with open(config_path, 'r') as fp:
                    print(f"Configuration File Loaded - {config_path}")
                    config = toml.load(fp)
                self._config_dir = Path(config_path).resolve().parent
            else:
                config = {
                    "basic": {
                        "save_file_dir": save_file_dir,
                        "default_task": default_task,
                        "default_website": default_website,
                        "crawler_mode": crawler_mode,
                        "crawler_max_steps": crawler_max_steps,
                    },
                    "agent": {
                        "input_info": input_info,
                        "grounding_strategy": grounding_strategy,
                        "max_auto_op": max_auto_op,
                        "max_continuous_no_op": max_continuous_no_op,
                        "highlight": highlight
                    },
                    "openai": {
                        "rate_limit": rate_limit,
                        "model": model,
                        "temperature": temperature
                    }
                }
                self._config_dir = Path(os.getcwd())
            config = config or {}

            basic_cfg = config.setdefault("basic", {})
            basic_cfg.setdefault("save_file_dir", save_file_dir)
            basic_cfg.setdefault("default_task", default_task)
            basic_cfg.setdefault("default_website", default_website)
            basic_cfg.setdefault("crawler_mode", crawler_mode)
            basic_cfg.setdefault("crawler_max_steps", crawler_max_steps)

            openai_cfg = config.setdefault("openai", {})
            openai_cfg.setdefault("rate_limit", rate_limit)
            openai_cfg.setdefault("model", model)
            openai_cfg.setdefault("temperature", temperature)

            # Normalize/augment config to a unified schema expected by this agent
            # Ensure an 'agent' section exists
            agent_cfg = config.setdefault("agent", {})
            agent_cfg.setdefault("input_info", agent_cfg.get("input_info", ["screenshot"]))
            agent_cfg.setdefault("grounding_strategy", agent_cfg.get("grounding_strategy", grounding_strategy))
            agent_cfg.setdefault("max_auto_op", agent_cfg.get("max_auto_op", max_auto_op))
            agent_cfg.setdefault("max_continuous_no_op", agent_cfg.get("max_continuous_no_op", max_continuous_no_op))
            agent_cfg.setdefault("highlight", agent_cfg.get("highlight", highlight))
            agent_cfg.setdefault("heuristic", agent_cfg.get("heuristic", True))
            self._capture_screenshots = "screenshot" in (agent_cfg.get("input_info") or [])

            # Map Playwright section to 'browser' section
            pw = config.get("playwright", {}) or {}
            pw_viewport = pw.get("viewport") or viewport
            if not isinstance(pw_viewport, dict):
                pw_viewport = viewport
            config["browser"] = {
                "headless": headless if headless is not None else False,
                "args": args or [],
                "browser_app": browser_app,
                "persistant": persistant,
                "persistant_user_path": persistant_user_path,
                "save_video": save_video,
                "viewport": pw_viewport,
                "tracing": tracing if tracing is not None else bool(pw.get("tracing", False)),
                "trace": pw.get("trace", trace),
            }

        except FileNotFoundError:
            print(f"Error: File '{os.path.abspath(config_path)}' not found.")
            
        except toml.TomlDecodeError:
            print(f"Error: File '{os.path.abspath(config_path)}' is not a valid TOML file.")

        self.worker_id = worker_id
        self.config_path = config_path
        self.config = config
        self._meta = {"config_dir": str(getattr(self, "_config_dir", Path(os.getcwd()))), "config_path": str(config_path) if config_path else None}
        self.complete_flag = False
        self.session_control = {
            'active_page': None,
            'context': None,
            'browser': None
        }
        # Browserbase session bookkeeping
        self._bb_session_id = None
        self._bb_api_key = None
        self.tasks = [self.config["basic"]["default_task"]]
        self._step_metrics: list[dict[str, float | int | bool]] = []
        self._manifest: ManifestRecord | None = None
        self._manifest_selectors: dict[str, Any] = {}
        self._manifest_step_used = False
        self._manifest_config = self.config.get("manifest", {}) or {}

        save_root = Path(self.config["basic"]["save_file_dir"]).resolve()
        save_root.mkdir(parents=True, exist_ok=True)
        self.main_path = self.config["basic"].get("main_path")
        #os.makedirs(self.main_path, exist_ok=True)
        print(f"SeeActAgent_{worker_id}: Saving files to {self.main_path}")
        self.action_space = ["CLICK", "PRESS ENTER", "HOVER", "SCROLL UP", "SCROLL DOWN", "NEW TAB", "CLOSE TAB",
                             "GO BACK", "GO FORWARD",
                             "TERMINATE", "SELECT", "TYPE", "GOTO", "MEMORIZE"]  # Define the list of actions here

        self.no_value_op = ["CLICK", "PRESS ENTER", "HOVER", "SCROLL UP", "SCROLL DOWN", "NEW TAB", "CLOSE TAB",
                            "PRESS HOME", "PRESS END", "PRESS PAGEUP", "PRESS PAGEDOWN"
                                                                       "GO BACK",
                            "GO FORWARD",
                            "TERMINATE", "NONE"]

        self.with_value_op = ["SELECT", "TYPE", "GOTO", "MEMORIZE", "SAY"]

        self.no_element_op = ["PRESS ENTER", "SCROLL UP", "SCROLL DOWN", "NEW TAB", "CLOSE TAB", "GO BACK", "GOTO",
                              "PRESS HOME", "PRESS END", "PRESS PAGEUP", "PRESS PAGEDOWN",
                              "GO FORWARD",
                              "TERMINATE", "NONE", "MEMORIZE", "SAY"]

        # Initialize the primary logger and the developer logger
        self.logger = self._setup_logger(redirect_to_dev_log=False)

        self.engine = None
        self.taken_actions = []

        if self.config["agent"]["grounding_strategy"] == "pixel_2_stage":
            self.prompts = self._initialize_prompts_pure_vision()
        self.prompts = self._initialize_prompts()
        self.time_step = 0
        self.valid_op = 0
        self.continuous_no_op = 0
        self.predictions = []
        self.visited_links = []
        self._page = None
        # LLM timeout (seconds) for one-turn decisions (configurable via [openai].timeout_sec)
        self._llm_timeout_sec = int((self.config.get("openai") or {}).get("timeout_sec", 20))
        # Final result payload if we reach completion (used by runner to emit task_complete with results)
        self.final_result: dict | None = None
        # Macro configuration (selectors/patterns) with sane defaults; may be overridden by TOML [macros]
        mcfg = (self.config.get("macros") or {})
        self._macro_cfg = {
            "product_href_patterns": mcfg.get("product_href_patterns", ["/products", "/product"]),
            "product_section_selectors": mcfg.get("product_section_selectors", ["main", "section[role='main']"]),
            "exclude_regions": mcfg.get("exclude_regions", {"bottom_fraction": 0.2, "top_fraction": 0.0}),
            "variant_labels": mcfg.get("variant_labels", ["size", "color", "variant", "style"]),
            "atc_selectors": mcfg.get("atc_selectors", [
                "button[name*='add' i]", "button:has-text('add to cart' i)", "form[action*='/cart/add'] button[type='submit']",
            ]),
            "checkout_selectors": mcfg.get("checkout_selectors", [
                "button:has-text('checkout' i)", "a[href*='/checkout']",
            ]),
        }
        

    @staticmethod
    def _categorize_url(url: str) -> Tuple[bool, bool]:
        """Return flags (is_collection_like, is_pdp_like) derived from a URL."""
        lowered = (url or "").lower()
        is_collection = any(token in lowered for token in (
            "/collections/",
            "/collection/",
            "/categories/",
            "/category/",
            "/catalog/",
            "/search",
        ))
        is_pdp = any(token in lowered for token in (
            "/products/",
            "/product/",
            "/item/",
            "/p/",
        ))
        return is_collection, is_pdp

    async def _generate_with_timeout(self, **kwargs):
        """Run self.engine.generate in a thread with a timeout to prevent step stalls."""
        def _call():
            #try:
            return self.engine.generate(**kwargs)
            #except Exception as e:
            #    return f"ELEMENT: Z\nACTION: NONE\nVALUE: None\nERROR: {e}"
        try:  # TODO: there seems to be an occasional bug where the timeout results in a new instance of the task being created?
            return await asyncio.wait_for(asyncio.to_thread(_call), timeout=self._llm_timeout_sec)
        except asyncio.TimeoutError:
            self.logger.warning(f"LLM generation timed out after {self._llm_timeout_sec} seconds", exc_info=True)  
            return None

    def _task_keywords(self) -> list[str]:
        """Extract light-weight task keywords (nouns-ish) from confirmed_task for generic biasing."""
        text = (self.tasks[-1] if self.tasks else "")
        text = text.lower()
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text)
        stop = {
            'the','and','for','from','with','into','onto','that','this','those','these','any','all','you','your',
            'use','open','find','page','product','products','add','cart','checkout','before','payment','stop','return',
            'choose','select','size','color','variant','best','now','click','go','to','of','in','on','at','it','then',
            'site','store','shop','default','one','two','by','rate','rating'
        }
        kws = [t for t in tokens if t not in stop and len(t) <= 24]
        seen = set(); out = []
        for k in kws:
            if k not in seen:
                seen.add(k); out.append(k)
        return out[:6]

    async def _macro_next_action(self, override=False):
        """Heuristic next action to progress common shopping flows (collection → PDP → size → ATC → checkout)."""
        #if not override and self.config["agent"].get("heuristic", True) is not True:
        #    return None
        if override:
            self.logger.warning("Heuristic macro action override in effect")
        elif self.config["agent"].get("heuristic", True) is not True:
            return None # disabled
        

        url = self.page.url # TODO: make sure this isn't none

        # Helper to build a prediction dict from a locator
        async def _pred_from_locator(loc, desc, tag="div"):
            return {
                "action_generation": "",
                "action_grounding": "",
                "element": {
                    "center_point": (0, 0),
                    "description": desc,
                    "tag_with_role": tag,
                    "box": [0, 0, 0, 0],
                    "selector": loc,
                    "tag": tag,
                },
                "action": "CLICK",
                "value": "",
            }
        manifest_selectors = getattr(self, "_manifest_selectors", {}) or {}

        # 1) If on collection page: click a product tile anchor under main grid, generic bias using task keywords; avoid header/footer
        try:
            if "/collections/" in url and "/products/" not in url:
                manifest_col = manifest_selectors.get("collections") or {}
                manifest_product = manifest_col.get("product_link")
                if manifest_product:
                    try:
                        loc = self.page.locator(manifest_product).first
                        if await loc.count() > 0 and await loc.is_visible(timeout=1000):
                            self._manifest_step_used = True
                            return await _pred_from_locator(loc, "Open product tile", "a")
                    except Exception:
                        pass
                # Build product anchor locator from macro config patterns
                href_or = " | ".join([f"a[href*='{p}']" for p in self._macro_cfg["product_href_patterns"]])
                sections = ", ".join(self._macro_cfg["product_section_selectors"])
                candidates = self.page.locator(f"{sections} {href_or}")
                count = await candidates.count()
                if count == 0:
                    candidates = self.page.locator(href_or)
                    count = await candidates.count()
                if count > 0:
                    # Score by visibility + position + small keyword bias from task
                    best = (0.0, None)
                    vp = await self.page.evaluate("() => ({w: window.innerWidth, h: window.innerHeight})")
                    keywords = self._task_keywords()
                    bf = float(self._macro_cfg.get("exclude_regions", {}).get("bottom_fraction", 0.2) or 0.2)
                    tf = float(self._macro_cfg.get("exclude_regions", {}).get("top_fraction", 0.0) or 0.0)
                    for i in range(min(count, 20)):
                        el = candidates.nth(i)
                        try:
                            if not await el.is_visible(timeout=500):
                                continue
                            box = await el.bounding_box() or {"y": 0, "height": 0}
                            y = box.get("y", 0) + box.get("height", 0) / 2
                            # exclude header/top and footer/bottom regions
                            if vp:
                                h = float(vp.get("h", 1000))
                                if y > (1.0 - bf) * h:
                                    continue
                                if y < tf * h:
                                    continue
                            txt = (await el.inner_text(timeout=500) or "").lower()
                            score = 1.0
                            # small bias if anchor text contains any task keyword
                            if keywords and any(k in txt for k in keywords):
                                score += 0.75
                            # prefer higher on page modestly
                            score += max(0.0, 0.8 - (y / max(1.0, float(vp.get("h", 1000)))))
                            if score > best[0]:
                                best = (score, el)
                        except Exception:
                            continue
                    if best[1] is not None:
                        return await _pred_from_locator(best[1], "Open product tile", "a")
        except Exception:
            pass
        # 2) On PDP: select a visible variant if required (generic labels/roles), or click Add to Cart if enabled
        is_collection_like, is_pdp_like = self._categorize_url(url or "")
        is_pdp = is_pdp_like
        if not is_pdp and not is_collection_like:
            try:
                if await self.page.locator('form[action*="/cart/add"]').count() > 0:
                    is_pdp = True
            except Exception:
                is_pdp = False
        if is_pdp:
            manifest_pdp = manifest_selectors.get("pdp") or {}
            manifest_variant = manifest_pdp.get("variant_widget")
            if manifest_variant:
                try:
                    loc = self.page.locator(manifest_variant).first
                    if await loc.count() > 0 and await loc.is_visible(timeout=1000):
                        self._manifest_step_used = True
                        return await _pred_from_locator(loc, f"Select variant {manifest_variant}", "button")
                except Exception:
                    pass
            # Try common role/inputs for variant selection generically
            variant_selectors = [
                "[role='radio']", "[role='option']", "input[type='radio'] + label", "select", "[role='combobox']",
                "button:has-text('size' i)", "button:has-text('variant' i)", "label:has-text('size' i)",
            ]
            for sel in variant_selectors:
                try:
                    loc = self.page.locator(sel).first
                    if await loc.count() > 0 and await loc.is_visible(timeout=1000):
                        return await _pred_from_locator(loc, f"Select variant {sel}", "button")
                except Exception:
                    continue
            # 3) Add to cart
            manifest_atc = manifest_pdp.get("add_to_cart")
            if manifest_atc:
                try:
                    loc = self.page.locator(manifest_atc).first
                    if await loc.count() > 0 and await loc.is_visible(timeout=1000):
                        self._manifest_step_used = True
                        return await _pred_from_locator(loc, "Add to cart", "button")
                except Exception:
                    pass
            atc_selectors = self._macro_cfg["atc_selectors"]
            for sel in atc_selectors:
                try:
                    loc = self.page.locator(sel).first
                    if await loc.count() > 0 and await loc.is_visible(timeout=1000):
                        return await _pred_from_locator(loc, "Add to cart", "button")
                except Exception:
                    continue
        # 4) Checkout in cart drawer
        checkout_sel = None
        try:
            # Prefer drawer dialogs
            dialog = self.page.locator('[role="dialog"]').first
            if await dialog.count() > 0:
                cta = dialog.locator('button:has-text("Checkout"), a[href*="/checkout"]').first
                if await cta.count() > 0 and await cta.is_visible(timeout=1000):
                    checkout_sel = cta
        except Exception:
            pass
        if checkout_sel is not None:
            return await _pred_from_locator(checkout_sel, "Checkout", "button")
        # 5) Fallback to Cart then Checkout (only if we likely added items)
        try:
            manifest_cart = manifest_selectors.get("cart") or {}
            manifest_checkout = manifest_cart.get("checkout")
            if manifest_checkout:
                try:
                    loc = self.page.locator(manifest_checkout).first
                    if await loc.count() > 0 and await loc.is_visible(timeout=1000):
                        self._manifest_step_used = True
                        return await _pred_from_locator(loc, "Checkout", "button")
                except Exception:
                    pass
            # Prefer clicking a cart icon/link if visible
            cart_icon = self.page.locator('a[href*="/cart"], button[aria-label*="cart" i], a[aria-label*="cart" i]').first
            if await cart_icon.count() > 0 and await cart_icon.is_visible(timeout=1000):
                return await _pred_from_locator(cart_icon, "Open Cart", "a")
            # Build absolute /cart URL based on current origin
            from urllib.parse import urlparse
            parsed = urlparse(url)
            origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else None
            if origin and "/cart" not in url:
                return {
                    "action_generation": "",
                    "action_grounding": "",
                    "element": None,
                    "action": "GOTO",
                    "value": origin + "/cart",
                }
            # Already on /cart → click checkout
            # Check for presence of line items before trying checkout
            has_items = False
            try:
                item_count = await self.page.locator('form[action*="/cart"] [name="updates[]"], .cart-item, .cart__items').count()
                has_items = item_count > 0
            except Exception:
                pass
            if has_items:
                # Try to adjust quantity to 2 if possible before checkout
                try:
                    qty_input = self.page.locator('form[action*="/cart"] input[name="updates[]"], input[type="number"][name*="update"]').first
                    if await qty_input.count() > 0 and await qty_input.is_visible(timeout=500):
                        return {
                            "action_generation": "",
                            "action_grounding": "",
                            "element": {
                                "center_point": (0, 0),
                                "description": "Quantity",
                                "tag_with_role": "input",
                                "box": [0, 0, 0, 0],
                                "selector": qty_input,
                                "tag": "input",
                            },
                            "action": "TYPE",
                            "value": "2",
                        }
                except Exception:
                    pass
                cart_checkout = self.page.locator('button:has-text("Checkout"), a[href*="/checkout"]').first
                if await cart_checkout.count() > 0 and await cart_checkout.is_visible(timeout=1000):
                    return await _pred_from_locator(cart_checkout, "Checkout", "button")
        except Exception:
            pass
        self.logger.debug("No heuristic macro action applicable")
        # If nothing found, return NO-OP to advance
        return {"action_generation": "", "action_grounding": "", "element": None, "action": "NONE", "value": ""}

    async def _maybe_complete_and_extract(self) -> dict | None:
        """Detect a generic completion (e.g., checkout) and extract results. Returns TERMINATE prediction if done."""
        url = self.page.url or ""
        if not url:
            self.logger.debug("No URL found in maybe_complete_and_extract")
            return None
        is_checkout = ("/checkout" in url) or ("checkout." in url)
        if not is_checkout:
            return None
        result: dict = {"products": [], "subtotal": None, "total": None, "checkout_url": url}
        #try:
            # Prefer an order summary region if present
        region = self.page.locator('[aria-label*="order summary" i], section:has-text("Order summary"), aside').first
        items = []
        if await region.count() > 0:
            li = region.locator("li, .product, .line-item, tr")
            n = min(await li.count(), 5)
            for i in range(n):
                node = li.nth(i)
                #try:
                txt = (await node.inner_text(timeout=500) or "").strip()
                if not txt:
                    continue
                title = txt.split("\n")[0][:80]
                m = re.search(r"[×x]\s*(\d+)", txt)
                qty = int(m.group(1)) if m else 1
                items.append({"title": title, "qty": qty})
                #except Exception:
                    #continue
            region_text = (await region.inner_text(timeout=800) or "")
            msub = re.search(r"Subtotal\s*([$€£]\s?[0-9][0-9,.]+)", region_text, re.I)
            mtot = re.search(r"Total\s*([$€£]\s?[0-9][0-9,.]+)", region_text, re.I)
            result["subtotal"] = msub.group(1) if msub else None
            result["total"] = mtot.group(1) if mtot else None
        if not items:
            cart_form = self.page.locator('form[action*="/cart"]').first
            if await cart_form.count() > 0:
                rows = cart_form.locator(".cart__row, tr, li, .cart-item")
                n = min(await rows.count(), 5)
                for i in range(n):
                    node = rows.nth(i)
                    #try:
                    txt = (await node.inner_text(timeout=500) or "").strip()
                    if not txt:
                        continue
                    title = txt.split("\n")[0][:80]
                    m = re.search(r"[×x]\s*(\d+)", txt)
                    qty = int(m.group(1)) if m else 1
                    items.append({"title": title, "qty": qty})
                    #except Exception:
                    #    continue
        result["products"] = items
        #except Exception:
        #    pass
        self.final_result = result
        self.logger.info(f"Completion detected. Result: {json.dumps(result, ensure_ascii=False)}")
        return {"action_generation": "", "action_grounding": "", "element": None, "action": "TERMINATE", "value": "STOP"}

    def _initialize_prompts(self):
        """Initialize prompt information including dynamic action space."""
        action_format = f"ACTION: Choose an action from allowed actions."  # Dynamically generate action_format based on self.action_space

        return {
            "system_prompt": '''You are assisting humans doing web navigation tasks step by step. At each stage, you can see the webpage by a screenshot and know the previous actions before the current step decided by yourself that have been executed for this task through recorded history. You need to decide on the first following action to take.''',

            "action_space": '''
Here are the descriptions of all allowed actions:

No Value Operations:
- CLICK: Click on a webpage element using the mouse.
- HOVER: Move the mouse over a webpage element without clicking.
- PRESS ENTER: Press the Enter key, typically to submit a form or confirm an input.
- SCROLL UP: Scroll the webpage upwards by half of the window height.
- SCROLL DOWN: Scroll the webpage downwards by half of the window height.
- PRESS HOME: Scroll to the top of the webpage.
- PRESS END: Scroll to the bottom of the webpage.
- PRESS PAGEUP: Scroll up by one window height.
- PRESS PAGEDOWN: Scroll down by one window height.
- CLOSE TAB: Close the current tab in the browser.
- NEW TAB: Open a new tab in the browser.
- GO BACK: Navigate to the previous page in the browser history.
- GO FORWARD: Navigate to the next page in the browser history.
- TERMINATE: End the current task, typically used when the task is considered complete or requires potentially harmful actions.
- NONE: Indicates that no action is necessary at this stage. Used to skip an action or wait.

With Value Operations:
- SELECT: Choose an option from a dropdown menu or <select> element. The value indicates the option to select.
- TYPE: Enter text into a text area or text box. The value is the text to be typed.
- GOTO: Navigate to a specific URL. The value is the URL to navigate to.
- SAY: Output answers or other information you want to tell the user.
- MEMORIZE: Keep some content into action history to memorize it.
''',

            "question_description": '''The screenshot below shows the webpage you see. Think step by step before outlining the next action step at the current stage. Clearly outline which element in the webpage users will operate with as the first next target element, its detailed location, and the corresponding operation.

To be successful, it is important to follow the following rules: 
1. You should only issue a valid action given the current observation. 
2. You should only issue one action at a time
3. For handling the select dropdown elements on the webpage, it's not necessary for you to provide completely accurate options right now. The full list of options for these elements will be supplied later.
4. Unlike humans, for typing (e.g., in text areas, text boxes) and selecting (e.g., from dropdown menus or <select> elements), you should try directly typing the input or selecting the choice, bypassing the need for an initial click. 
5. You should not attempt to create accounts, log in or do the final submission. 
6. Terminate when you deem the task complete or if it requires potentially harmful actions.
7. Do not generate same action as the previous one, try different ways if keep failing
8. When there is a floating banner like ads, login, or survey floating taking more than 30% of the page, close the floating banner to proceed, the close button could look like a x on the right top corner, or choose NO THANKS to close it.
9. When there is a floating banner on top or bottom of the page like cookie policy taking less than 30% of the page, ignore the banner to proceed.  
10. After typing text into search or text input area, the next action is normally PRESS ENTER
11. When there are bouding boxes in the screenshot, interact with the elements in the bounding boxes
12. When there are multiple clickable buttons having the same value, choose the one with less obstacles in the screenshot.
''',

            "referring_description": f"""(Reiteration)
First, reiterate your next target element, its detailed location, and the corresponding operation.

(Multichoice Question)
Below is a multi-choice question, where the choices are elements in the webpage. All elements are arranged in the order based on their height on the webpage, from top to bottom (and from left to right). This arrangement in addition to the normalized coordinates can be used to locate them. From the screenshot, find out where and what each one is on the webpage, taking into account both their text content and HTML details. Then, determine whether one matches your target element if your action involves an element. Please examine the choices one by one. Choose the matching one. If multiple options match your answer, choose the most likely one by re-examining the screenshot, the choices, and your further reasoning.""",

            "element_format": '''(Final Answer)
Finally, conclude your answer using the format below. Ensure your answer is strictly adhering to the format provided below. Please do not leave any explanation in your answers of the final standardized format part, and this final part should be clear and certain. The element choice, action, and value should be in three separate lines.

Format:

ELEMENT: The uppercase letter of your choice.''',

            "action_format": action_format,  # Use the dynamically generated action_format

            "value_format": '''VALUE: Provide additional input based on ACTION. (If it doesn't involve a value, write "None"'''
        }

    def _initialize_prompts_pure_vision(self):
        """Specifically for Vision-only agents"""

        return {
            "system_prompt": '''You are assisting humans doing web navigation tasks step by step. At each stage, you can see the webpage by a screenshot and know the previous actions before the current step decided by yourself that have been executed for this task through recorded history. You need to decide on the first following action to take.''',

            "question_description": '''
    The screenshot below shows the webpage you see. Think step by step before outlining the next action step at the current stage. Clearly outline which element in the webpage users will operate with as the first next target element, its detailed location, and the corresponding operation.

To be successful, it is important to follow the following rules: 
1. You should only issue a valid action given the current observation. 
2. You should only issue one action at a time
4. Unlike humans, for typing (e.g., in text areas, text boxes), you should try directly typing the input, bypassing the need for an initial click. 
5. You should not attempt to create accounts, log in or do the final submission. 
6. Terminate when you deem the task complete or if it requires potentially harmful actions.
7. Do not generate same action as the previous one, try different ways if keep failing
8. When there is a floating banner like ads, login, or survey floating taking more than 30% of the page, close the floating banner to proceed, the close button could look like a x on the right top corner, or choose NO THANKS to close it.
9. When there is a floating banner on top or bottom of the page like cookie policy taking less than 30% of the page, ignore the banner to proceed.  
10. After typing text into search or text input area, the next action is normally PRESS ENTER
11. When there are bouding boxes in the screenshot, interact with the elements in the bounding boxes
12. When there are multiple clickable buttons having the same value, choose the one with less obstacles in the screenshot.            
                
                
    Here are the descriptions of all allowed actions:

    No Value Operations:
    - CLICK: Click on a webpage element using the mouse.
    - HOVER: Move the mouse over a webpage element without clicking.
    - PRESS ENTER: Press the Enter key, typically to submit a form or confirm an input.
    - SCROLL UP: Scroll the webpage upwards by half of the window height.
    - SCROLL DOWN: Scroll the webpage downwards by half of the window height.
    - PRESS HOME: Scroll to the top of the webpage.
    - PRESS END: Scroll to the bottom of the webpage.
    - PRESS PAGEUP: Scroll up by one window height.
    - PRESS PAGEDOWN: Scroll down by one window height.
    - CLOSE TAB: Close the current tab in the browser.
    - NEW TAB: Open a new tab in the browser.
    - GO BACK: Navigate to the previous page in the browser history.
    - GO FORWARD: Navigate to the next page in the browser history.
    - TERMINATE: End the current task, typically used when the task is considered complete or requires potentially harmful actions.
    - NONE: Indicates that no action is necessary at this stage. Used to skip an action or wait.

    With Value Operations:
    - TYPE: Enter text into a text area or text box. The value is the text to be typed.
    - GOTO: Navigate to a specific URL. The value is the URL to navigate to.
    - SAY: Output answers or other information you want to tell the user.
    - MEMORIZE: Keep some content into action history to memorize it.
    
    Finally, conclude your answer using the format below. Ensure your answer is strictly adhering to the format provided below. Please do not leave any explanation in your answers of the final standardized format part, and this final part should be clear and certain. The element choice, action, and value should be in three separate lines.
    
    Format:

    ELEMENT: The description about the exact location to help locating the exact position to operate at. Required for click and type.
    
    ACTION: Choose an action from allowed actions.
                
    VALUE: Provide additional input based on ACTION. (If it doesn't involve a value, write "None"            
                
    '''}

    def update_action_space(self, new_actions):
        """Update the action space and regenerate the action_format prompt."""
        if isinstance(new_actions, list) and all(isinstance(item, str) for item in new_actions):
            self.action_space = new_actions
            self.prompts["action_format"] = f"ACTION: Choose an action from {{{', '.join(self.action_space)}}}."
        else:
            print("Invalid action space provided. It must be a list of strings.")

    def _setup_logger(self, redirect_to_dev_log=False):
        """Set up a logger to log to both file and console within the main_path."""
        logger_name = f'SeeActAgent_Worker_{self.worker_id}'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:  # Avoid adding handlers multiple times
            # Create a file handler for writing logs to a file
            log_filename = f'agent_worker_{self.worker_id}.log'
            f_handler = logging.FileHandler(os.path.join(self.main_path, log_filename), mode='a')
            f_handler.setLevel(logging.INFO)

            # Create a console handler for printing logs to the terminal
            c_handler = logging.StreamHandler()
            c_handler.setLevel(logging.INFO)

            # Create formatters for file and console handlers
            file_formatter = logging.Formatter('%(asctime)s - %(message)s')
            console_formatter = logging.Formatter('%(message)s')

            # Set formatters for file and console handlers
            f_handler.setFormatter(file_formatter)
            c_handler.setFormatter(console_formatter)

            # Add the handlers to the logger
            logger.addHandler(f_handler)
            if not redirect_to_dev_log:  # Only add console handler if not redirecting to dev log
                logger.addHandler(c_handler)
                
        return logger

    async def page_on_close_handler(self):
        # Corrected to use 'self' for accessing class attributes
        if self.session_control['context']:
            try:
                await self.page.title()
            except:
                self.logger.info(
                    "The active tab was closed. Will switch to the last page (or open a new default google page)")
                try:
                    ctx = self.session_control['context']
                    if ctx and ctx.pages:
                        self.page = ctx.pages[-1]
                        await self.page.bring_to_front()
                        self.logger.info(f"Switched the active tab to: {self.page.url}")
                except Exception:
                    pass

    def save_action_history(self, filename="action_history.txt"):
        """Save the history of taken actions to a file in the main path."""
        history_path = os.path.join(self.main_path, filename)
        with open(history_path, 'w') as f:
            for action in self.taken_actions:
                f.write(action + '\n')
        self.logger.info(f"Action history saved to: {history_path}")

    async def page_on_navigation_handler(self, frame):
        # Corrected to use 'self' for accessing class attributes
        self.page = frame.page

    async def page_on_crash_handler(self, page):
        # Corrected logging method
        self.logger.info(f"Page crashed: {page.url}")
        self.logger.info("Try to reload")
        await page.reload()

    async def page_on_open_handler(self, page):
        # Added 'self' to the handler functions to reference the current instance of the class
        page.on("framenavigated", self.page_on_navigation_handler)
        page.on("close", self.page_on_close_handler)
        page.on("crash", self.page_on_crash_handler)
        self.page = page
        # Additional event listeners can be added here
        try:
            if self.config["agent"]["grounding_strategy"] == "text_choice_som":
                with open(os.path.join(dirname(__file__), "mark_page.js")) as f:
                    mark_page_script = f.read()
                await page.wait_for_load_state("domcontentloaded")
                await self.page.evaluate(mark_page_script)
                #await self.session_control['active_page'].evaluate(mark_page_script)
        except Exception as e:
            if "Execution context was destroyed" in str(e):
                # Navigation happened again — just skip
                self.logger.warning("Skipped script injection due to page reload (context destroyed) at page_on_open_handler")
            else:
                self.logger.error(f"Failed to set up page scripts: {e}")
                raise e

    async def start(self, headless=None, args=None, website=None):
        if self.engine is None: # TODO: at this point, this probably would never happen
            try:
                if engine_factory is None:
                    from .demo_utils.inference_engine import engine_factory as _ef  # type: ignore
                else:
                    _ef = engine_factory
                self.engine = _ef(**self.config['openai'])
            except Exception as e:
                # Defer engine errors to first use to allow construction under tests without API keys
                self.logger.warning("LLM Engine Initialization failed.")
                raise e
        if website:
            self._load_manifest_for_url(website)
        # Lazy import to respect test stubs in sys.modules
        pa = importlib.import_module("playwright.async_api")
        ap = pa.async_playwright()
        try:
            # Preferred pattern
            self.playwright = await ap.start()
        except Exception:
            self.logger.error("Failed to start Playwright... trying fallback", exc_info=True)
            try:
                # Support context manager style stubs
                async with ap as _pw:
                    self.playwright = _pw
            except Exception:
                # Fallback: use the object directly if it looks like a Playwright
                self.playwright = ap
        # Runtime provider (local vs CDP/browserbase)
        runtime = self.config.get("runtime", {}) or {}
        provider = str(runtime.get("provider", "local")).lower()
        cdp_url = runtime.get("cdp_url")
        headers = runtime.get("headers")
        if isinstance(cdp_url, str):
            cdp_url = os.path.expandvars(cdp_url)
        # Normalize headers: pass dict when possible; retry with array fallback if needed
        headers_out = None
        if isinstance(headers, dict):
            headers_out = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in headers.items()} or None
        elif isinstance(headers, list):
            headers_out = headers or None

        # Resolve Browserbase session if requested
        if provider == "browserbase" and not cdp_url and bb_create is not None:
            # Read project_id and optional api_base
            project_id = runtime.get("project_id") or os.getenv("BROWSERBASE_PROJECT_ID")
            api_base = runtime.get("api_base") or os.getenv("BROWSERBASE_API_BASE")
            session_options = runtime.get("session_options") or {}
            # Expand environment references if provided like "${VAR}"
            if isinstance(project_id, str):
                project_id = os.path.expandvars(project_id)
            if isinstance(api_base, str):
                api_base = os.path.expandvars(api_base)
            # Expand any env vars within session_options recursively
            def _expand(v):
                import os as _os
                if isinstance(v, str):
                    return _os.path.expandvars(v)
                if isinstance(v, dict):
                    return {k: _expand(val) for k, val in v.items()}
                if isinstance(v, list):
                    return [_expand(x) for x in v]
                return v
            if isinstance(session_options, dict):
                session_options = _expand(session_options)
            try:
                pid, api_key = bb_resolve(project_id, os.getenv("BROWSERBASE_API_KEY"))  # type: ignore
                cdp_url, session_id = bb_create(pid, api_key, api_base=api_base, session_options=session_options)  # type: ignore
                self._bb_session_id = session_id
                self._bb_api_key = api_key
                #try:
                self.logger.info(f"Browserbase session created: session_id={session_id} cdp_url={cdp_url}")
                #except Exception:
                    #pass
            except Exception as e:
                raise RuntimeError(f"Failed to create Browserbase session: {e}")

        if provider in ("cdp", "browserbase") and cdp_url:
            try:
                self.session_control['browser'] = await self.playwright.chromium.connect_over_cdp(cdp_url, headers=headers_out)
            except Exception:
                self.logger.warning("CDP connection with dict headers failed... retrying with array format")
                # Fallback to array of {name,value}
                headers_array = None
                if isinstance(headers, dict):
                    headers_array = [{"name": k, "value": os.path.expandvars(v) if isinstance(v, str) else v} for k, v in headers.items()] or None
                elif isinstance(headers, list):
                    headers_array = headers or None
                self.session_control['browser'] = await self.playwright.chromium.connect_over_cdp(cdp_url, headers=headers_array)
            # Record call for test stubs that expect chromium.calls
            #try:            
            if not hasattr(self.playwright.chromium, "calls"):
                setattr(self.playwright.chromium, "calls", [])
            hdr = headers_out if headers_out is not None else (headers_array if 'headers_array' in locals() else {})
            self.playwright.chromium.calls.append((cdp_url, hdr or {}))
            #except Exception:
            #    pass
            # Use an existing remote context if available, otherwise create one
            ctx = None
            try:
                if getattr(self.session_control['browser'], 'contexts', None):
                    if self.session_control['browser'].contexts:
                        ctx = self.session_control['browser'].contexts[0]
            except Exception:
                ctx = None
            if ctx is None:
                ctx = await normal_new_context_async(self.session_control['browser'],
                                                     viewport=self.config['browser']['viewport'])
            self.session_control['context'] = ctx
        else:
            self.session_control['browser'] = await normal_launch_async(
                self.playwright,
                headless=self.config['browser']['headless'] if headless is None else headless,
                args=self.config['browser']['args'] if args is None else args
            )
            self.session_control['context'] = await normal_new_context_async(self.session_control['browser'],
                                                                             viewport=self.config['browser']['viewport'])

        assert self.session_control['context'] is not None
        assert self.session_control['browser'] is not None
        
        self.session_control['context'].on("page", self.page_on_open_handler)
        p = await self.session_control['context'].new_page()
        # Ensure viewport emulation is set when connecting over CDP (may default to None)
        #try:
        if not p.viewport_size:
            await p.set_viewport_size(self.config['browser']['viewport'])
        #except Exception:
        #    pass

        # Optional crawler_mode: start tracing only when explicitly enabled
        if bool((self.config.get("basic") or {}).get("crawler_mode", False)):
            await self.session_control['context'].tracing.start(screenshots=True, snapshots=True)

        target_url = self.config['basic']['default_website'] if website is None else website
        try:
            # Prefer direct page handle to avoid relying on event handler under tests
            self.page = self.page or p
            await (self.page or p).goto(target_url, wait_until="domcontentloaded")
            self.logger.info(f"Loaded website: {target_url}")
        except Exception as e:
            self.logger.info("Failed to fully load the webpage before timeout")
            self.logger.info(e)

            # await asyncio.sleep(2)

    def update_prompt_part(self, part_name, new_text):
        """Update the specified part of the prompt information."""
        if part_name in self.prompts:
            self.prompts[part_name] = new_text
            return True
        else:
            print(f"Prompt part '{part_name}' not found.")
            return False

    @staticmethod
    def _extract_domain(url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        try:
            host = urlparse(url).hostname
            if host and host.startswith("www."):
                host = host[4:]
            return host
        except Exception:
            return None

    def _load_manifest_for_url(self, url: Optional[str]) -> None:
        self._manifest = None
        self._manifest_selectors = {}
        if not url:
            return
        domain = self._extract_domain(url)
        manifest_cfg = self._manifest_config or {}
        manifest_dir_value = manifest_cfg.get("dir") or manifest_cfg.get("cache_dir")
        manifest_dir_path = None
        if manifest_dir_value:
            manifest_dir_path = Path(os.path.expandvars(str(manifest_dir_value))).resolve()
        candidates = []
        if domain:
            candidates.append(domain)
            parts = domain.split(".")
            if len(parts) > 2:
                candidates.append(".".join(parts[-2:]))
        for cand in candidates:
            if manifest_dir_path is None:
                continue
            manifest = load_manifest_from_dir(cand, manifest_dir_path)
            if manifest:
                self._manifest = manifest
                selectors = manifest.selectors
                self._manifest_selectors = selectors
                overlay_selector = (selectors.get("overlays") or {}).get("close_button") if isinstance(selectors, dict) else None
                if overlay_selector:
                    register_overlay_hint(cand, overlay_selector)
                break

    def _manifest_prompt_hint(self) -> Optional[str]:
        selectors = getattr(self, "_manifest_selectors", {}) or {}
        if not selectors:
            return None
        hints = []
        search_sel = (selectors.get("search") or {}).get("input")
        if search_sel:
            hints.append(f"Use CSS selector `{search_sel}` for the search input.")
        product_sel = (selectors.get("collections") or {}).get("product_link")
        if product_sel:
            hints.append(f"Product tiles use `{product_sel}`.")
        variant_sel = (selectors.get("pdp") or {}).get("variant_widget")
        if variant_sel:
            hints.append(f"Variant widgets use `{variant_sel}`.")
        atc_sel = (selectors.get("pdp") or {}).get("add_to_cart")
        if atc_sel:
            hints.append(f"Add to cart button is `{atc_sel}`.")
        checkout_sel = (selectors.get("cart") or {}).get("checkout")
        if checkout_sel:
            hints.append(f"Checkout CTA uses `{checkout_sel}`.")
        if not hints:
            return None
        return "Manifest hints:\n- " + "\n- ".join(hints)

    def generate_prompt(self, task=None, previous=None, choices=None):

        """Generate a prompt based on the current task, previous actions, and choices."""
        # assert task is not None, "Please input the task."

        prompt_list = []
        if self.config["agent"]["grounding_strategy"] == "pixel_2_stage":
            system_prompt_input = self.prompts["system_prompt"]
            question_description_input = self.prompts["question_description"]
            previous_ = self.taken_actions if self.taken_actions else None
            prompt_list.extend(
                generate_new_query_prompt(system_prompt=system_prompt_input,
                                          task=self.tasks[-1], previous_actions=previous_,
                                          question_description=question_description_input))
            return prompt_list
        else:

            system_prompt_input = self.prompts["system_prompt"]
            action_space_input = self.prompts["action_space"]
            question_description_input = self.prompts["question_description"]
            referring_input = self.prompts["referring_description"]
            element_format_input = self.prompts["element_format"]
            action_format_input = self.prompts["action_format"]
            value_format_input = self.prompts["value_format"]

            manifest_hint = self._manifest_prompt_hint()
            if manifest_hint:
                question_description_input = question_description_input + "\n\n" + manifest_hint

            # print(previous)

            previous_ = self.taken_actions if self.taken_actions else None

            # print(previous_)

            prompt_list.extend(
                generate_new_query_prompt(system_prompt=system_prompt_input + "\n" + action_space_input,
                                          task=self.tasks[-1], previous_actions=previous_,
                                          question_description=question_description_input))
            prompt_list.append(
                generate_new_referring_prompt(referring_description=referring_input,
                                              element_format=element_format_input,
                                              action_format=action_format_input, value_format=value_format_input,
                                              choices=choices))

            return prompt_list

    async def perform_action(self, target_element=None, action_name=None, value=None, target_coordinates=None,
                             element_repr=None):
        # Repeat/no-progress guard: suppress repeated CLICK on the same URL and nudge scroll
        try:
            _url_now = self.page.url
        except Exception:
            _url_now = ""
        _target_key = f"{action_name}|{element_repr or ''}".strip()
        if getattr(self, "_last_url", None) is None:
            self._last_url = ""
        if getattr(self, "_last_target_key", None) is None:
            self._last_target_key = ""
        if getattr(self, "_repeat_clicks", None) is None:
            self._repeat_clicks = 0
        if _url_now == self._last_url and _target_key and _target_key == self._last_target_key and action_name == "CLICK":
            self._repeat_clicks += 1
        else:
            self._repeat_clicks = 0
        if self._repeat_clicks >= 1 and action_name == "CLICK":
            # If we've already nudged once and still repeating, escalate to macro fallback
            if self._repeat_clicks >= 2:
                try:
                    macro_pred = await self._macro_next_action()
                    # Execute macro action immediately to break the loop
                    exec_msg = await self.perform_action(
                        target_element=macro_pred.get("element"),
                        action_name=macro_pred.get("action"),
                        value=macro_pred.get("value"),
                        target_coordinates=None,
                        element_repr=(macro_pred.get("element") or {}).get("description") if macro_pred.get("element") else None,
                    )
                    return exec_msg
                except Exception as _e:
                    pass
            # Otherwise, do a nudge scroll and rescan next step
            #try:
            await self.page.evaluate("window.scrollBy(0, Math.min(window.innerHeight * 0.6, 600));")
            #except Exception:
            #    pass
            auto_msg = f"Auto-nudge: suppressed repeat of {action_name} {element_repr}; scrolled"
            self.taken_actions.append(auto_msg)
            # Update guard state and return without executing the repeated action
            self._last_url = _url_now
            self._last_target_key = _target_key
            return auto_msg

        if self.config["agent"]["grounding_strategy"] == "pixel_2_stage":
            selector = "pixel_coordinates"
        if target_element is not None:
            selector = target_element['selector']
            element_repr = target_element['description']
        else:
            selector = None


        page = self.page

        if action_name == "CLICK" and selector:
            if selector == "pixel_coordinates":
                delay = random.randint(50, 150)
                await self.page.mouse.click(round(target_coordinates["x"]), round(target_coordinates["y"]), delay=delay)
            else:
                await selector.click(timeout=2000)
                self.logger.info(f"Clicked on element: {element_repr}")
        elif action_name == "HOVER" and selector:

            if selector == "pixel_coordinates":
                delay = random.randint(50, 150)
                await self.page.mouse.hover(round(target_coordinates["x"]), round(target_coordinates["y"]), delay=delay)
            else:
                await selector.hover(timeout=2000)
                self.logger.info(f"Hovered over element: {element_repr}")



        elif action_name == "TYPE" and selector:

            if selector == "pixel_coordinates":
                delay = random.randint(50, 150)
                await self.page.mouse.click(round(target_coordinates["x"]), round(target_coordinates["y"]), delay=delay)

                await self.page.keyboard.press("Control+A")
                await self.page.keyboard.press("Backspace")
                # value = stringfy_value(action['fill_text'])
                await self.page.keyboard.type(value)
            else:
                await selector.fill(value)
                self.logger.info(f"Typed '{value}' into element: {element_repr}")

        elif action_name == "SCROLL UP":
            await page.evaluate(f"window.scrollBy(0, -{self.config['browser']['viewport']['height'] // 2});")
            self.logger.info("Scrolled up")
        elif action_name == "SCROLL DOWN":
            await page.evaluate(f"window.scrollBy(0, {self.config['browser']['viewport']['height'] // 2});")
            self.logger.info("Scrolled down")
        elif action_name == "PRESS HOME":
            await page.keyboard.press('Home')
            self.logger.info("Pressed Home key")
        elif action_name == "PRESS END":
            await page.keyboard.press('End')
            self.logger.info("Pressed End key")
        elif action_name == "PRESS PAGEUP":
            await page.keyboard.press('PageUp')
            self.logger.info("Pressed PageUp key")
        elif action_name == "PRESS PAGEDOWN":
            await page.keyboard.press('PageDown')
            self.logger.info("Pressed PageDown key")
        elif action_name == "NEW TAB":
            new_page = await self.session_control['context'].new_page()
            self.session_control['active_page'] = new_page  # TODO: why was this not here originally?
            # self.session_control['pages'].append(new_page)
            self.logger.info("Opened a new tab")
        elif action_name == "CLOSE TAB":
            await page.close()
            self.logger.info("Closed the current tab")
        elif action_name == "GO BACK":
            await page.go_back()
            self.logger.info("Navigated back")
        elif action_name == "GO FORWARD":
            await page.go_forward()
            self.logger.info("Navigated forward")
        elif action_name == "GOTO" and value:
            # Normalize to absolute URL if relative
            try:
                if not re.match(r"^[a-zA-Z]+://", str(value)):
                    from urllib.parse import urlparse
                    cur = urlparse(page.url)
                    origin = f"{cur.scheme}://{cur.netloc}" if cur.scheme and cur.netloc else ""
                    value = origin + str(value)
            except Exception:
                self.logger.warning("Failed to parse current URL for GOTO normalization: "+ str(page.url))
            await page.goto(value, wait_until="domcontentloaded")
            self.logger.info(f"Navigated to {value}")
        elif action_name == "PRESS ENTER" and selector:
            if selector == "pixel_coordinates":
                delay = random.randint(50, 150)
                await self.page.mouse.click(round(target_coordinates["x"]), round(target_coordinates["y"]), delay=delay)
            await selector.press('Enter')
            self.logger.info(f"Pressed Enter on element: {element_repr}")
        elif action_name == "PRESS ENTER":
            await page.keyboard.press('Enter')
            self.logger.info(f"Pressed Enter on element: {element_repr}")
        elif action_name == "SELECT" and selector:
            await select_option(selector, value)
            self.logger.info(f"Selected option '{value}' from element: {element_repr}")
        elif action_name == "TERMINATE":
            self.complete_flag = True
            self.logger.info("Task has been marked as complete. Terminating...")
        elif action_name in ["NONE"]:
            self.logger.info("No action necessary at this stage. Skipped")
        elif action_name in ["SAY"]:
            self.logger.info(f"Say {value} to the user")
        elif action_name in ["MEMORIZE"]:
            self.logger.info(f"Keep {value} to the action history.")
        else:
            raise NotImplementedError(f"Unsupported or improperly specified action: {action_name}")
        if action_name in self.no_element_op and target_element is None:
            new_action = action_name
        else:
            if selector == "pixel_coordinates":
                new_action = element_repr + " -> " + action_name
            else:
                new_action = "[" + target_element['tag_with_role'] + "]" + " "
                new_action += target_element['description'] + " -> " + action_name
        if action_name in self.with_value_op:
            new_action += ": " + value

        # self.dev_logger.info(new_action)
        # Update repeat guard state and append action for runner parity
        #try:
        self._last_url = self.page.url
        #except Exception:
        #    pass
        self._last_target_key = _target_key
        # Runner path calls perform_action directly; ensure actions are recorded
        #try:
        self.taken_actions.append(new_action)
        #except Exception:
        #    pass
        return new_action

    async def predict(self):

        """
        Generate a prediction for the next action based on the webpage elements and previous actions.
        """

        self.time_step += 1
        #try:
        # originally this was looked for in session_control['active_page']
        await self.page.wait_for_load_state('domcontentloaded')
        # Auto-dismiss overlays (cookie banners, modals) to surface targets
        #except Exception:
        #   pass
        await auto_dismiss_overlays(self.page, max_clicks=2)

        # Completion gate: if on checkout (or equivalent), extract results and terminate
        try:
            maybe_done = await self._maybe_complete_and_extract()
            if maybe_done:
                self.predictions.append(maybe_done)
                return maybe_done
        except Exception:
            pass
        scan_t0 = time.time()
        elements = await get_interactive_elements_with_playwright(
            self.page, self.config['browser']['viewport']
        )
        scan_ms = int((time.time() - scan_t0) * 1000)

        '''
             0: center_point =(x,y)
             1: description
             2: tag_with_role: tag_head with role and type # TODO: Consider adding more
             3. box
             4. selector
             5. tag
             '''

        elements = sorted(elements, key=lambda el: (
            el["center_point"][1], el["center_point"][0]))  # Sorting by y and then x coordinate

        elements = [{**x, "idx": i, "option": generate_option_name(i)} for i, x in enumerate(elements)]

        # In crawler mode, get random link and click on it
        if bool((self.config.get("basic") or {}).get("crawler_mode", False)):
            if self.time_step > self.config["basic"]["crawler_max_steps"]:
                self.logger.info("Crawler reached max steps, going to stop")
                self.complete_flag = True
                return None

            links = [x for x in elements if x['tag_with_role'] == 'a']
            random_link = get_random_link(links)
            while random_link in self.visited_links and len(links) > 0:
                random_link = get_random_link(links)
            if random_link is None:
                return None

            prediction = {"action_generation": "Random chosen link", "action_grounding": "Random chosen link",
                          "element": random_link,
                          "action": "CLICK", "value": 'None'}
            self.predictions.append(prediction)
            self.visited_links.append(random_link)
            self.logger.info(prediction)
            await self.take_screenshot()
            await self.start_playwright_tracing()
            return prediction

        #try:
        if self.config["agent"]["grounding_strategy"] == "text_choice_som":
            with open(os.path.join(dirname(__file__), "mark_page.js")) as f:
                mark_page_script = f.read()
            await self.page.evaluate(mark_page_script)
            await self.page.evaluate("unmarkPage()")
            await self.page.evaluate("""elements => {
                return window.som.drawBoxes(elements);
                }""", elements)
        #except Exception as e:
        #    self.logger.info(f"Mark page script error {e}")

        # Generate choices for the prompt (batched for parity with demo)
        include_dom = bool((self.config.get("experiment") or {}).get("include_dom_in_choices", False))
        exp_cfg = self.config.get("experiment", {}) or {}
        top_k = int(exp_cfg.get("top_k", 50))
        fixed_choice_batch_size = int(exp_cfg.get("fixed_choice_batch_size", 22))
        dynamic_choice_batch_size = int(exp_cfg.get("dynamic_choice_batch_size", 0))
        if top_k > 0:
            elements = elements[:top_k]

        screenshot_path = await self.take_screenshot()
        #await self.take_screenshot()

        terminal_width = 10
        self.logger.info(f"Step - {self.time_step}\n{'-' * terminal_width}\nAction Generation ->")
        self.logger.info("TASK: " + self.tasks[-1])
        self.logger.info("Previous:")
        for action in self.taken_actions:
            self.logger.info(action)
        self.logger.info("-" * terminal_width)
        self.logger.info("One-turn mode: generating structured decision in a single call.")
        self.logger.info("-" * terminal_width)
        self.logger.info(f"Grounding Strategy: {self.config['agent']['grounding_strategy']}")
        if self.config["agent"]["grounding_strategy"] == "pixel_2_stage":
            # For pixel mode, keep current behavior minimal (not commonly used here)
            pred_element_label, pred_action, pred_value = postprocess_action_lmm_pixel("")
            prediction = {"action_generation": "", "action_grounding": "", "element": None,
                          "action": pred_action, "value": pred_value, "coordinates": None,
                          "description": pred_element_label}
            self._step_metrics.append({
                "scan_ms": scan_ms,
                "llm_ms": 0,
                "macro_used": False,
                "num_candidates": 0,
                "manifest_available": bool(getattr(self, "_manifest_selectors", {})),
                "manifest_used": getattr(self, "_manifest_step_used", False),
            })

        else:
            num_choices = len(elements)
            manifest_selectors = getattr(self, "_manifest_selectors", {}) or {}
            self._manifest_step_used = False
            step_metrics_entry = {
                "scan_ms": scan_ms,
                "llm_ms": 0,
                "macro_used": False,
                "num_candidates": num_choices,
                "manifest_available": bool(manifest_selectors),
                "manifest_used": False,
            }
            self.logger.info(f"Found {num_choices} candidate elements on the page.")
            # Heuristic-first: try a macro step immediately if it can make obvious progress
            try:
                self.logger.info("Trying heuristic macro action before LLM grounding...")
                macro_pred = await self._macro_next_action()
                if macro_pred is None:
                    pass # macro next action disable
                elif macro_pred.get("action") not in [None, "", "NONE"]:
                    self.predictions.append(macro_pred)
                    step_metrics_entry["macro_used"] = True
                    step_metrics_entry["manifest_used"] = getattr(self, "_manifest_step_used", False)
                    self._step_metrics.append(step_metrics_entry)
                    return macro_pred
                else:
                    self.logger.info("Heuristic macro action did not yield a valid action.")
            except Exception as e:
                self.logger.error("Heuristic macro action failed for unknown reasons", e)
                raise e
            if num_choices == 0:
                self.logger.warning("No interactive elements found; using fallback macro action, overriding settings.")
                prediction = await self._macro_next_action(override=True)
                self.predictions.append(prediction)
                step_metrics_entry["macro_used"] = True
                step_metrics_entry["manifest_used"] = getattr(self, "_manifest_step_used", False)
                self._step_metrics.append(step_metrics_entry)
                return prediction
            if dynamic_choice_batch_size > 0:
                step_length = max(1, int(dynamic_choice_batch_size))
            else:
                step_length = max(1, int(min(num_choices, fixed_choice_batch_size)))
            prediction = None
            llm_ms_total = 0
            for start in range(0, num_choices, step_length):
                end = min(num_choices, start + step_length)
                batch = elements[start:end]
                choices = format_choices(batch, include_dom=include_dom)
                options = format_options(choices)
                choice_text = f"Action Grounding ->" + "\n" + options
                for line in choice_text.split('\n'):
                    self.logger.info(line)
                prompt = self.generate_prompt(task=self.tasks[-1], previous=self.taken_actions, choices=choices)
                t_llm0 = time.time()
                output = await self._generate_with_timeout(
                    prompt=prompt,
                    image_path=screenshot_path,
                    turn_number=1,
                    ouput_0="",
                )
                llm_ms_total += int((time.time() - t_llm0) * 1000)
                step_metrics_entry["llm_ms"] = llm_ms_total
                self.logger.info(f"[llm] Action Grounding Output [llm]")
                if output:
                    for line in output.split('\n'):
                        self.logger.info(line.strip())

                if not output:
                    # LLM timed out → use fallback macro for this step
                    prediction = await self._macro_next_action()
                    step_metrics_entry["macro_used"] = True
                    self.logger.info(f"metrics: scan_ms={scan_ms} llm_ms={llm_ms_total} macro_used=true num_candidates={num_choices}")
                    step_metrics_entry["manifest_used"] = getattr(self, "_manifest_step_used", False)
                    self._step_metrics.append(step_metrics_entry)
                    self.predictions.append(prediction)
                    return prediction

                pred_element_label, pred_action, pred_value = postprocess_action_lmm(output)
                local_idx = get_index_from_option_name(pred_element_label) if len(pred_element_label) in [1, 2] else None
                if local_idx is not None and 0 <= local_idx < len(batch) and pred_action.strip() in [
                    "CLICK", "SELECT", "TYPE", "PRESS ENTER", "HOVER", "TERMINATE"]:
                    pred_element = batch[local_idx]
                    prediction = {"action_generation": "", "action_grounding": output, "element": pred_element,
                                  "action": pred_action, "value": pred_value}
                    break
                elif pred_action.strip() in ["PRESS ENTER", "TERMINATE"]:
                    prediction = {"action_generation": "", "action_grounding": output, "element": None,
                                  "action": pred_action, "value": pred_value}
                    break
            if prediction is None:
                # No actionable choice from batches → use fallback macro
                self.logger.warning("LLM did not select a valid element from any batch; using fallback macro action.")
                prediction = await self._macro_next_action(override=True)
                step_metrics_entry["macro_used"] = True
            try:
                self.logger.info(f"metrics: scan_ms={scan_ms} llm_ms={llm_ms_total} macro_used={step_metrics_entry['macro_used']} num_candidates={num_choices}")
            except Exception:
                pass
            step_metrics_entry["manifest_used"] = getattr(self, "_manifest_step_used", False)
            self._step_metrics.append(step_metrics_entry)

        self.predictions.append(prediction)

        # return {"action_generation": output0, "action_grounding": output, "element": pred_element,
        #         "action": pred_action, "value": pred_value}

        return prediction

        # return output0,output,pred_element, pred_action, pred_value

    async def execute(self, prediction_dict):
        """
        Execute the predicted action on the webpage.
        """

        if prediction_dict is None:
            self.complete_flag = True
            return

        #try:
        # Clear the marks before action
        if self.config["agent"]["grounding_strategy"] == "text_choice_som":
            await self.page.evaluate("unmarkPage()")
        #except Exception as e:
        #    pass

        pred_element = prediction_dict["element"]
        pred_action = prediction_dict["action"]
        pred_value = prediction_dict["value"]
        pred_coordinate = None
        pred_element_description=None
        if "description" in prediction_dict:
            pred_element_description=prediction_dict["description"]
        if self.config["agent"]["grounding_strategy"] == "pixel_2_stage":
            pred_coordinate = prediction_dict["coordinates"]

        try:
            if (pred_action not in self.no_element_op) and pred_element == None:
                # self.dev_logger.info
                self.logger.info("DEBUG: WHAT IS PRED ACTION???:" + pred_action)
                # self.dev_logger.info("DEBUG WHAT IS self.no_element_op???:"+ self.no_element_op)
                pred_action = "NONE"
            new_action = await self.perform_action(pred_element, pred_action, pred_value, pred_coordinate,pred_element_description)
            self.taken_actions.append(new_action)
            if pred_action != "NONE":
                self.valid_op += 1
                self.continuous_no_op = 0
            else:
                self.continuous_no_op += 1
            if bool((self.config.get("basic") or {}).get("crawler_mode", False)):
                await self.stop_playwright_tracing()
                await self.save_traces()

            return 0
        except Exception as e:

            new_action = f"Failed to perform {pred_action} on {pred_element['description']} with value '{pred_value}': {e}"

            traceback_info = traceback.format_exc()
            error_message = f"Error executing action {pred_action}: {str(e)}"
            print(traceback_info)
            error_message_with_traceback = f"{error_message}\n\nTraceback:\n{traceback_info}"

            self.logger.info(new_action)
            self.taken_actions.append(new_action)
            self.continuous_no_op += 1
            return 1

    async def stop(self):

        try:
            close_context = self.session_control['context']
            self.session_control['context'] = None
            if close_context:
                await close_context.close()
                self.logger.info("Browser context closed.")
        except Exception as e:
            # Quiet teardown failures
            self.logger.info(f"Context close skipped: {e}")

        final_json = {"task": self.tasks, "website": self.config["basic"]["default_website"],
                      "num_step": len(self.taken_actions), "action_history": self.taken_actions}

        def locator_serializer(obj):
            """Convert non-serializable objects to a serializable format."""
            try:
                from playwright.async_api import Locator as _Locator  # type: ignore
                if isinstance(obj, _Locator):
                    return str(obj)
            except Exception:
                pass
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        # Using the custom default function in json.dump
        with open(os.path.join(self.main_path, 'all_predictions.json'), 'w', encoding='utf-8') as f:
            json.dump(self.predictions, f, default=locator_serializer, indent=4)

        with open(os.path.join(self.main_path, 'result.json'), 'w', encoding='utf-8') as file:
            json.dump(final_json, file, indent=4)
        self.logger.info("Agent stopped.")

        saveconfig(self.config, os.path.join(self.main_path, 'config.toml'))

        # Close Browserbase session if we created one
        if self._bb_session_id and bb_close is not None and self._bb_api_key:
            bb_close(self._bb_session_id, self._bb_api_key)  # type: ignore
            self.logger.info("Browserbase session closed.")
        # Stop Playwright
        if getattr(self, "playwright", None):
            await self.playwright.stop()
            self.logger.info("Playwright closed.")


    def clear_action_history(self):
        """
        Clears the history of actions taken by the agent.
        """
        self.taken_actions.clear()
        self.logger.info("Cleared action history.")

    def reset_comlete_flag(self, flag=False):
        self.complete_flag = flag

    def change_task(self, new_task, clear_history=False):
        """
        Changes the task requirement for the agent.

        Parameters:
        - new_task: The new task requirement as a string.
        """
        if new_task and isinstance(new_task, str):

            self.logger.info(f"Changed task from {self.tasks[-1]} to: {new_task}")
            self.tasks.append(new_task)
            # Optionally clear action history when changing task
            if clear_history:
                self.clear_action_history()
            else:
                self.taken_actions.append(f"Changed task from {self.tasks[-2]} to: {new_task}")

        else:
            self.logger.info("Invalid new task. It must be a non-empty string.")

        # Optionally, you can save the taken_actions to a file or database for record-keeping

    # ADD no op count and op count, add limit to op

    # decompose run to predict and execute.

    async def take_screenshot(self):
        if not self._capture_screenshots:
            return None
        path = self.screenshot_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            await self.page.screenshot(path=path)
        except Exception as e:
            self.logger.warning(f"Failed to take screenshot: {e}")
            return None
        self.logger.debug(f"Screenshot saved to {path}")
        return path

    async def start_playwright_tracing(self):
        await self.session_control['context'].tracing.start_chunk(
            title=f'Step-{self.time_step}',
            name=f"{self.time_step}"
        )

    async def stop_playwright_tracing(self):
        await self.session_control['context'].tracing.stop_chunk(path=self.trace_path)

    async def save_traces(self):
        # Capture the DOM tree
        dom_tree = await self.page.evaluate("document.documentElement.outerHTML")
        os.makedirs(os.path.join(self.main_path, 'dom'), exist_ok=True)
        with open(self.dom_tree_path, 'w', encoding='utf-8') as f:
            f.write(dom_tree)

        # Capture the Accessibility Tree
        accessibility_tree = await self.page.accessibility.snapshot()
        os.makedirs(os.path.join(self.main_path, 'accessibility'), exist_ok=True)
        with open(self.accessibility_tree_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(accessibility_tree, indent=4))

    @property
    def page(self):
        if self._page is None:
            self._page = self.session_control['active_page']
        return self._page

    @page.setter
    def page(self, value):
        self._page = value

    @property
    def screenshot_path(self):
        return os.path.join(self.main_path, 'screenshots', f'screen_{self.worker_id}_{self.time_step}.png')

    @property
    def trace_path(self):
        return os.path.join(self.main_path, 'playwright_traces', f'{self.time_step}.zip')

    @property
    def dom_tree_path(self):
        return os.path.join(self.main_path, 'dom', f'{self.time_step}.html')

    @property
    def accessibility_tree_path(self):
        return os.path.join(self.main_path, 'accessibility', f'{self.time_step}.json')
