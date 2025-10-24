import re
import asyncio
import copy
from difflib import SequenceMatcher
from urllib.parse import urlparse
try:
    from playwright.sync_api import Playwright, expect, sync_playwright  # type: ignore
except Exception:  # optional for type hints only
    class Playwright:  # type: ignore
        pass
    def expect(*args, **kwargs):  # type: ignore
        return None
    def sync_playwright():  # type: ignore
        raise RuntimeError("sync_playwright unavailable in this environment")
# from playwright.async_api import async_playwright
from pathlib import Path
try:
    import toml  # type: ignore
except Exception:  # optional dependency for saveconfig
    class _TomlStub:  # type: ignore
        @staticmethod
        def dump(obj, fp):
            try:
                fp.write("# TOML output unavailable (toml not installed)\n")
            except Exception:
                pass
    toml = _TomlStub()  # type: ignore
from typing import List, Optional
import os

_OVERLAY_CACHE: dict[str, str] = {}


def _normalize_host(value: str) -> str:
    value = (value or "").lower()
    if value.startswith("http://") or value.startswith("https://"):
        try:
            value = urlparse(value).hostname or value
        except Exception:
            return value
    if value.startswith("www."):
        value = value[4:]
    return value


def register_overlay_hint(domain: str, selector: Optional[str]):  # pragma: no cover - small helper
    if not selector:
        return
    key = _normalize_host(domain)
    if key:
        _OVERLAY_CACHE[key] = selector


def _clear_overlay_cache():  # pragma: no cover - only used in tests
    _OVERLAY_CACHE.clear()


async def normal_launch_async(playwright: Playwright, headless=False, args=None):
    browser = await playwright.chromium.launch(
        traces_dir=None,
        headless=headless,
        args=args,
        # ignore_default_args=ignore_args,
        # chromium_sandbox=False,
    )
    return browser



async def normal_new_context_async(
        browser,
        storage_state=None,
        har_path=None,
        video_path=None,
        tracing=False,
        trace_screenshots=False,
        trace_snapshots=False,
        trace_sources=False,
        locale=None,
        geolocation=None,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
        viewport: dict = {"width": 1280, "height": 720},
):
    context = await browser.new_context(
        storage_state=storage_state,
        user_agent=user_agent,
        viewport=viewport,
        locale=locale,
        record_har_path=har_path,
        record_video_dir=video_path,
        geolocation=geolocation,
    )

    if tracing:
        await context.tracing.start(screenshots=trace_screenshots, snapshots=trace_snapshots, sources=trace_sources)
    return context

#
# def persistent_launch(playwright: Playwright, user_data_dir: str = ""):
#     context = playwright.chromium.launch_persistent_context(
#         user_data_dir=user_data_dir,
#         headless=False,
#         args=["--no-default-browser-check",
#               "--no_sandbox",
#               "--disable-blink-features=AutomationControlled",
#               ],
#         ignore_default_args=ignore_args,
#         user_agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
#         viewport={"width": 1280, "height": 720},
#         bypass_csp=True,
#         slow_mo=1000,
#         chromium_sandbox=True,
#         channel="chrome-dev"
#     )
#     return context

#
# async def persistent_launch_async(playwright: Playwright, user_data_dir: str = "", record_video_dir="video"):
#     context = await playwright.chromium.launch_persistent_context(
#         user_data_dir=user_data_dir,
#         headless=False,
#         args=[
#             "--disable-blink-features=AutomationControlled",
#         ],
#         ignore_default_args=ignore_args,
#         user_agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
#         # viewport={"width": 1280, "height": 720},
#         record_video_dir=record_video_dir,
#         channel="chrome-dev"
#         # slow_mo=1000,
#     )
#     return context



def remove_extra_eol(text):
    # Replace EOL symbols
    text = text.replace('\n', ' ')
    return re.sub(r'\s{2,}', ' ', text)


def get_first_line(s):
    first_line = s.split('\n')[0]
    tokens = first_line.split()
    if len(tokens) > 8:
        return ' '.join(tokens[:8]) + '...'
    else:
        return first_line

async def get_element_description(element, tag_name, role_value, type_value):
    '''
         Asynchronously generates a descriptive text for a web element based on its tag type.
         Handles various HTML elements like 'select', 'input', and 'textarea', extracting attributes and content relevant to accessibility and interaction.
    '''
    # text_content = await element.inner_text(timeout=0)
    # text = (text_content or '').strip()
    #
    # print(text)
    salient_attributes = [
        "alt",
        "aria-describedby",
        "aria-label",
        "aria-role",
        "input-checked",
        # "input-value",
        "label",
        "name",
        "option_selected",
        "placeholder",
        "readonly",
        "text-value",
        "title",
        "value",
    ]

    parent_value = "parent_node: "
    parent_locator = element.locator('xpath=..')
    num_parents = await parent_locator.count()
    if num_parents > 0:
        # only will be zero or one parent node
        parent_text = (await parent_locator.inner_text(timeout=0) or "").strip()
        if parent_text:
            parent_value += parent_text
    parent_value = remove_extra_eol(get_first_line(parent_value)).strip()
    if parent_value == "parent_node:":
        parent_value = ""
    else:
        parent_value += " "

    if tag_name == "select":
        text1 = "Selected Options: "
        text3 = " - Options: "

        text2 = await element.evaluate(
            "select => select.options[select.selectedIndex].textContent", timeout=0
        )

        if text2:
            options = await element.evaluate("select => Array.from(select.options).map(option => option.text)",
                                             timeout=0)
            text4 = " | ".join(options)

            if not text4:
                text4 = await element.text_content(timeout=0)
                if not text4:
                    text4 = await element.inner_text(timeout=0)


            return parent_value+text1 + remove_extra_eol(text2.strip()) + text3 + text4

    input_value = ""

    none_input_type = ["submit", "reset", "checkbox", "radio", "button", "file"]

    if tag_name == "input" or tag_name == "textarea":
        if role_value not in none_input_type and type_value not in none_input_type:
            text1 = "input value="
            text2 = await element.input_value(timeout=0)
            if text2:
                input_value = text1 + "\"" + text2 + "\"" + " "

    text_content = await element.text_content(timeout=0)
    text = (text_content or '').strip()

    # print(text)
    if text:
        text = remove_extra_eol(text)
        if len(text) > 80:
            text_content_in = await element.inner_text(timeout=0)
            text_in = (text_content_in or '').strip()
            if text_in:
                return input_value + remove_extra_eol(text_in)
        else:
            return input_value + text

    # get salient_attributes
    text1 = ""
    for attr in salient_attributes:
        attribute_value = await element.get_attribute(attr, timeout=0)
        if attribute_value:
            text1 += f"{attr}=" + "\"" + attribute_value.strip() + "\"" + " "

    text = (parent_value + text1).strip()
    if text:
        return input_value + remove_extra_eol(text.strip())


    # try to get from the first child node
    first_child_locator = element.locator('xpath=./child::*[1]')

    num_childs = await first_child_locator.count()
    if num_childs>0:
        for attr in salient_attributes:
            attribute_value = await first_child_locator.get_attribute(attr, timeout=0)
            if attribute_value:
                text1 += f"{attr}=" + "\"" + attribute_value.strip() + "\"" + " "

        text = (parent_value + text1).strip()
        if text:
            return input_value + remove_extra_eol(text.strip())

    return None


async def get_element_data(element, tag_name,viewport_size,seen_elements=[]):
    try:
        tag_name_list = ['a', 'button',
                         'input',
                         'select', 'textarea', 'adc-tab']


        if await element.is_hidden(timeout=0) or await element.is_disabled(timeout=0):
            return None



        rect = await element.bounding_box() or {'x': -1, 'y': -1, 'width': 0, 'height': 0}


        if rect['x']<0 or rect['y']<0 or rect['width']<=4 or rect['height']<=4 or rect['y']+rect['height']>viewport_size["height"] or rect['x']+ rect['width']>viewport_size["width"]:
            return None

        box_raw = [rect['x'], rect['y'], rect['width'], rect['height']]
        box_model = [rect['x'], rect['y'], rect['x'] + rect['width'], rect['y'] + rect['height']]
        center_point = (round((box_model[0] + box_model[2]) / 2 / viewport_size["width"], 3),
                        round((box_model[1] + box_model[3]) / 2 / viewport_size["height"], 3))

        if center_point in seen_elements:
            return None

        # await aprint(element,tag_name)

        if tag_name in tag_name_list:
            tag_head = tag_name
            real_tag_name = tag_name
        else:
            real_tag_name = await element.evaluate("element => element.tagName.toLowerCase()", timeout=0)
            if real_tag_name in tag_name_list:
                # already detected
                return None
            else:
                tag_head = real_tag_name

        text_element = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'td', "div","em","center","strong","b","i","small","mark","abbr","cite","q","blockquote","span","nobr"]

        # Fetch role/type early to allow interactive text-like nodes via ARIA or onclick
        role_value = await element.get_attribute('role', timeout=0)
        type_value = await element.get_attribute('type', timeout=0)

        if real_tag_name in text_element:
            interactive_roles = {"button", "link", "menuitem", "tab", "checkbox", "radio", "switch", "option"}
            has_onclick = await element.get_attribute('onclick', timeout=0)
            if (role_value and role_value.lower() in interactive_roles) or has_onclick:
                # treat as interactive though tag is generic
                pass
            else:
                return None
        # await aprint("start to get element description",element,tag_name )
        description = await get_element_description(element, real_tag_name, role_value, type_value)
        # print(description)
        if not description:
            return None

        if role_value:
            tag_head += " role=" + "\"" + role_value + "\""
        if type_value:
            tag_head += " type=" + "\"" + type_value + "\""

        '''
                     0: center_point =(x,y)
                     1: description
                     2: tag_with_role: tag_head with role and type # TODO: Consider adding more
                     3. box
                     4. selector
                     5. tag
                     '''
        selector = element
        outer_html = None
        try:
            outer_html = await element.evaluate("el => el.outerHTML", timeout=0)
            if outer_html:
                # sanitize and truncate to keep prompts lean
                outer_html = remove_extra_eol(outer_html)
                if len(outer_html) > 600:
                    outer_html = outer_html[:600] + "..."
        except Exception:
            outer_html = None

        return {
            "center_point":center_point,
            "description":description,
            "tag_with_role":tag_head,
            "box_raw":box_raw,
            "box":box_model,
            "selector":selector,
            "tag":real_tag_name,
            "outer_html": outer_html,
        }
        # return [center_point, description, tag_head, box_model, selector, real_tag_name]
    except Exception as e:
        # print(e)
        return None

async def get_interactive_elements_js(page, viewport_size):
    js = """
    () => {
      const selectors = [
        'a', 'button', 'input', 'select', 'textarea',
        '[role="slider"]', '[role="option"]',
        '[role="button"]', '[role="link"]', '[role="menuitem"]',
        '[role="tab"]', '[role="checkbox"]', '[role="radio"]',
        '[onclick]', '[tabindex]'
      ];

      const seen = new Set();
      const results = [];
      const vw = window.innerWidth;
      const vh = window.innerHeight;

      function buildXPath(el) {
        if (el.id) return `//*[@id="${el.id}"]`;
        const parts = [];
        while (el && el.nodeType === Node.ELEMENT_NODE) {
          let index = 1;
          let sibling = el.previousSibling;
          while (sibling) {
            if (sibling.nodeType === Node.ELEMENT_NODE && sibling.nodeName === el.nodeName) {
              index++;
            }
            sibling = sibling.previousSibling;
          }
          const tagName = el.nodeName.toLowerCase();
          const part = `${tagName}[${index}]`;
          parts.unshift(part);
          el = el.parentNode;
        }
        return '/' + parts.join('/');
      }

      function cleanHTML(html) {
        if (!html) return null;
        html = html.replace(/\\s+/g, ' ').trim();
        return html.length > 600 ? html.slice(0, 600) + '...' : html;
      }

      function addElement(el, selOverride=null) {
        const rect = el.getBoundingClientRect();
        if (rect.width <= 4 || rect.height <= 4) return;
        if (rect.x < 0 || rect.y < 0 || rect.bottom > vh || rect.right > vw) return;

        const description = el.innerText?.trim() || el.getAttribute('aria-label') || el.getAttribute('alt') || '';
        if (!description) return;

        const cx = rect.x + rect.width / 2;
        const cy = rect.y + rect.height / 2;
        const normX = +(cx / vw).toFixed(3);
        const normY = +(cy / vh).toFixed(3);
        const key = `${normX}-${normY}`;
        if (seen.has(key)) return;
        seen.add(key);

        const role = el.getAttribute('role');
        const type = el.getAttribute('type');
        const tag = el.tagName.toLowerCase();
        let tag_head = tag;
        if (role) tag_head += ` role="${role}"`;
        if (type) tag_head += ` type="${type}"`;

        results.push({
          center_point: [normX, normY],
          description,
          tag_with_role: tag_head,
          box_raw: [rect.x, rect.y, rect.width, rect.height],
          box: [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height],
          selector: selOverride || buildXPath(el),
          tag,
          outer_html: cleanHTML(el.outerHTML)
        });
      }

      // Primary pass
      for (const sel of selectors) {
        document.querySelectorAll(sel).forEach(el => addElement(el));
      }

      // Heuristic: clickable ancestors of images
      const imgs = Array.from(document.querySelectorAll('img, picture, figure')).slice(0, 200);
      for (const img of imgs) {
        const anchor = img.closest('a,[role="button"],[role="link"],[onclick]');
        if (!anchor) continue;
        addElement(anchor, 'ancestor-of-image');
      }

      return results;
    }
    """
    elements = await page.evaluate(js)
    return elements



async def get_interactive_elements_with_playwright(page, viewport_size=None):
    # Broaden initial pass to include common ARIA widgets that appear as non-form tags
    interactive_elements_selectors = [
        'a', 'button',
        'input',
        'select', 'textarea',
        '[role="slider"]',
        '[role="option"]',
    ]

    seen_elements = set()
    tasks = []


    for selector in interactive_elements_selectors:
        locator = page.locator(selector)
        element_count = await locator.count()
        for index in range(element_count):
            element = locator.nth(index)
            tag_name = selector
            task = get_element_data(element, tag_name,viewport_size)

            tasks.append(task)

    results = await asyncio.gather(*tasks)

    interactive_elements = []
    for i in results:
        if i:
            if i["center_point"] in seen_elements:
                continue
            else:
                seen_elements.add(i["center_point"])
                interactive_elements.append(i)

    # Heuristic pass: include clickable ancestors of prominent images/figures
    # Many PLPs render product tiles where the clickable target is an ancestor of an image
    try:
        img_like_selectors = ["img", "picture", "figure"]
        # Limit the number of img-like nodes to keep the scan bounded
        max_imgs = 200
        for sel in img_like_selectors:
            locator = page.locator(sel)
            count = await locator.count()
            # Only iterate over a capped number to avoid heavy scans
            for index in range(min(count, max_imgs)):
                el = locator.nth(index)
                try:
                    # Ensure the image is visible and within viewport bounds via its bounding box
                    bbox = await el.bounding_box() or {}
                    if not bbox or bbox.get("width", 0) <= 4 or bbox.get("height", 0) <= 4:
                        continue
                    # Prefer closest anchor; otherwise any ancestor that behaves like a link/button
                    anc_anchor = el.locator("xpath=ancestor::a[1]")
                    anc_clickable = el.locator(
                        "xpath=ancestor-or-self::*[@role='link' or @role='button' or @onclick][1]"
                    )
                    target_locator = anc_anchor if (await anc_anchor.count()) > 0 else anc_clickable
                    if (await target_locator.count()) == 0:
                        continue
                    target = target_locator.first
                    # De-duplicate by center_point using get_element_data
                    data = await get_element_data(target, "a", viewport_size, seen_elements)
                    if data:
                        interactive_elements.append(data)
                except Exception:
                    continue
    except Exception:
        # Fallback silently if this heuristic fails
        pass

    # Narrow the broad sweep to common interactive fallbacks to avoid scanning the entire DOM
    interactive_elements_selectors = [
        '[role="button"]',
        '[role="link"]',
        '[role="menuitem"]',
        '[role="tab"]',
        '[role="checkbox"]',
        '[role="radio"]',
        '[onclick]',
        '[tabindex]'
    ]
    tasks = []

    for selector in interactive_elements_selectors:
        locator = page.locator(selector)
        element_count = await locator.count()
        for index in range(element_count):
            element = locator.nth(index)
            tag_name = selector
            task = get_element_data(element, tag_name, viewport_size,seen_elements)

            tasks.append(task)

    results = await asyncio.gather(*tasks)


    for i in results:
        if i:
            if i["center_point"] in seen_elements:
                continue
            else:
                seen_elements.add(i["center_point"])
                interactive_elements.append(i)

    return interactive_elements


async def select_option(selector, value):
    best_option = [-1, "", -1]
    for i in range(await selector.locator("option").count()):
        option = await selector.locator("option").nth(i).inner_text()
        similarity = SequenceMatcher(None, option, value).ratio()
        if similarity > best_option[2]:
            best_option = [i, option, similarity]
    await selector.select_option(index=best_option[0], timeout=10000)
    return remove_extra_eol(best_option[1]).strip()


def saveconfig(config, save_file):
    """
    config is a dictionary.
    save_path: saving path include file name.
    """


    if isinstance(save_file, str):
        save_file = Path(save_file)
    if isinstance(config, dict):
        with open(save_file, 'w') as f:
            config_copy = copy.deepcopy(config)
            if "openai" in config_copy and isinstance(config_copy["openai"], dict):
                config_copy["openai"]["api_key"] = "Your API key here"
            toml.dump(config_copy, f)
    else:
        os.system(" ".join(["cp", str(config), str(save_file)]))


_OVERLAY_SELECTORS: List[str] = [
    '[role="dialog"] [aria-label*="close" i]',
    '[role="dialog"] [aria-label*="dismiss" i]',
    '[role="dialog"] button:has-text("Close")',
    '[role="dialog"] button:has-text("No thanks")',
    '[role="dialog"] button:has-text("Not now")',
    '[role="dialog"] button:has-text("Dismiss")',
    '[role="dialog"] button:has-text("Accept")',
    '[role="dialog"] button:has-text("Accept All")',
    '[role="dialog"] button:has-text("Agree")',
    '[role="dialog"] button:has-text("Allow all")',
    '[aria-label*="close" i]',
    '[aria-label*="dismiss" i]',
    'button:has-text("Close")',
    'button:has-text("No thanks")',
    'button:has-text("Not now")',
    'button:has-text("Dismiss")',
    'button:has-text("Accept")',
    'button:has-text("x")',
    'button:has-text("Accept All")',
    'button:has-text("Agree")',
    'button:has-text("Allow all")',
    'text=/^\s*[✖×X]\s*$/',
]


async def auto_dismiss_overlays(page, max_clicks: int = 3) -> int:
    """Try to dismiss common overlays/popups (cookie banners, newsletter modals).

    Heuristics:
    - Close/dismiss buttons by aria-label/text
    - Elements within role=dialog
    - Common text buttons: Close, No thanks, Not now, Dismiss, ×
    Returns the number of clicks performed.
    """
    clicked = 0
    try:
        parsed = urlparse(getattr(page, "url", ""))
        host = parsed.hostname or ""
    except Exception:
        host = ""
    host = _normalize_host(host)
    cached_selector = _OVERLAY_CACHE.get(host) if host else None

    selectors: List[str] = []
    if cached_selector:
        selectors.append(cached_selector)
    selectors.extend([s for s in _OVERLAY_SELECTORS if s != cached_selector])

    for _ in range(max_clicks):
        acted = False
        for sel in selectors:
            try:
                loc = page.locator(sel)
                count = await loc.count()
                if count == 0:
                    continue
                # Pick the first visible candidate
                for i in range(min(3, count)):
                    cand = loc.nth(i)
                    try:
                        if await cand.is_visible(timeout=500):
                            # Do not dismiss cart drawers/overlays that contain checkout/cart text
                            try:
                                dialog_parent = cand.locator("xpath=ancestor::*[@role='dialog'][1]")
                                if await dialog_parent.count() > 0:
                                    txt = (await dialog_parent.inner_text(timeout=500) or "").lower()
                                    if any(k in txt for k in ["checkout", "your cart", "cart subtotal", "shopping cart"]):
                                        continue
                            except Exception:
                                pass
                            await cand.click(timeout=1500)
                            clicked += 1
                            acted = True
                            if host:
                                _OVERLAY_CACHE[host] = sel
                            break
                    except Exception:
                        continue
                if acted:
                    break
            except Exception:
                continue
        if not acted:
            break
        # Give the DOM a moment to settle
        try:
            await page.wait_for_load_state('load', timeout=2500)
        except Exception:
            pass
    return clicked
