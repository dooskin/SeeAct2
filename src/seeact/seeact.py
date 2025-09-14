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
This script leverages the GPT-4V API and Playwright to create a web agent capable of autonomously performing tasks on webpages.
It utilizes Playwright to create browser and retrieve interactive elements, then apply [SeeAct Framework](https://osu-nlp-group.github.io/SeeAct/) to generate and ground the next operation.
The script is designed to automate complex web interactions, enhancing accessibility and efficiency in web navigation tasks.
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
import time

# Load .env from repo root if available (ergonomics)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
except Exception:
    pass

# TOML compatibility: prefer stdlib tomllib (Py3.11+), fallback to toml package
try:
    import tomllib as _toml_lib  # Python 3.11+
    def _load_toml(path_or_fp):
        import io
        if isinstance(path_or_fp, (str, bytes, os.PathLike)):
            with open(path_or_fp, 'rb') as f:
                return _toml_lib.load(f)
        # Ensure binary file-like for tomllib
        if hasattr(path_or_fp, 'read') and not isinstance(path_or_fp.read(0), bytes):
            # Convert text file to binary by reopening
            if hasattr(path_or_fp, 'name'):
                with open(path_or_fp.name, 'rb') as f:
                    return _toml_lib.load(f)
            raise TypeError('tomllib requires a binary file object')
        return _toml_lib.load(path_or_fp)
    _TomlDecodeError = getattr(_toml_lib, 'TOMLDecodeError', Exception)
except Exception:
    _toml_lib = None
    try:
        import toml as _toml_pkg  # External dependency
        def _load_toml(path_or_fp):
            return _toml_pkg.load(path_or_fp)
        _TomlDecodeError = getattr(_toml_pkg, 'TomlDecodeError', Exception)
    except Exception:  # No TOML parser available
        _toml_pkg = None
        def _load_toml(path_or_fp):
            raise ImportError("No TOML parser available. Install 'toml' or use Python 3.11+ (tomllib).")
        _TomlDecodeError = Exception
# Optional: torch is only needed when a local ranker is enabled. Defer import to avoid hard dep.
try:
    import torch  # type: ignore
except Exception:  # torch not installed; ranking will be disabled unless provided by env
    torch = None  # type: ignore
try:
    from aioconsole import ainput, aprint  # type: ignore
except Exception:
    async def ainput(prompt: str = "") -> str:
        return await asyncio.to_thread(input, prompt)

    async def aprint(*args, **kwargs) -> None:
        print(*args, **kwargs)
from playwright.async_api import async_playwright

from seeact.data_utils.format_prompt_utils import get_index_from_option_name, generate_option_name
from seeact.data_utils.prompts import generate_prompt
from seeact.data_utils.format_prompt_utils import format_options
from seeact.demo_utils.browser_helper import (normal_launch_async, normal_new_context_async,
                                       get_interactive_elements_with_playwright, select_option, saveconfig,
                                       auto_dismiss_overlays)
from seeact.demo_utils.format_prompt import format_choices, format_ranking_input, postprocess_action_lmm
from seeact.demo_utils.inference_engine import OpenaiEngine
# Browserbase API client (optional)
try:
    from seeact.runtime.browserbase_client import resolve_credentials as bb_resolve, create_session as bb_create, close_session as bb_close
except Exception:
    bb_resolve = bb_create = bb_close = None  # type: ignore
# Lazy-import ranking to avoid hard deps on torch when unused
from seeact.demo_utils.website_dict import website_dict
from difflib import SequenceMatcher

# Remove Huggingface internal warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class SessionControl:
    pages = []
    cdp_sessions = []
    active_page = None
    active_cdp_session = None
    context = None
    browser = None


session_control = SessionControl()


#
# async def init_cdp_session(page):
#     cdp_session = await page.context.new_cdp_session(page)
#     await cdp_session.send("DOM.enable")
#     await cdp_session.send("Overlay.enable")
#     await cdp_session.send("Accessibility.enable")
#     await cdp_session.send("Page.enable")
#     await cdp_session.send("Emulation.setFocusEmulationEnabled", {"enabled": True})
#     return cdp_session


async def page_on_close_handler(page):
    # print("Closed: ", page)
    if session_control.context:
        # if True:
        try:
            await session_control.active_page.title()
            # print("Current active page: ", session_control.active_page)
        except:
            await aprint("The active tab was closed. Will switch to the last page (or open a new default google page)")
            # print("All pages:")
            # print('-' * 10)
            # print(session_control.context.pages)
            # print('-' * 10)
            if session_control.context.pages:
                session_control.active_page = session_control.context.pages[-1]
                await session_control.active_page.bring_to_front()
                await aprint("Switched the active tab to: ", session_control.active_page.url)
            else:
                await session_control.context.new_page()
                try:
                    await session_control.active_page.goto("https://www.google.com/", wait_until="load")
                except Exception as e:
                    pass
                await aprint("Switched the active tab to: ", session_control.active_page.url)


async def page_on_navigatio_handler(frame):
    session_control.active_page = frame.page
    # print("Page navigated to:", frame.url)
    # print("The active tab is set to: ", frame.page.url)


async def page_on_crash_handler(page):
    await aprint("Page crashed:", page.url)
    await aprint("Try to reload")
    page.reload()


async def page_on_open_handler(page):
    # print("Opened: ",page)
    page.on("framenavigated", page_on_navigatio_handler)
    page.on("close", page_on_close_handler)
    page.on("crash", page_on_crash_handler)
    session_control.active_page = page
    # print("The active tab is set to: ", page.url)
    # print("All pages:")
    # print('-'*10)
    # print(session_control.context.pages)
    # print("active page: ",session_control.active_page)
    # print('-' * 10)


async def main(config, base_dir) -> None:
    # basic settings
    is_demo = config["basic"]["is_demo"]
    ranker_path = None
    try:
        ranker_path = config["basic"]["ranker_path"]
        if not os.path.exists(ranker_path):
            ranker_path = None
    except:
        pass
    # Use a boolean flag to control ranking; avoid mutating the path value later
    ranker_enabled = bool(ranker_path)

    save_file_dir = os.path.join(base_dir, config["basic"]["save_file_dir"]) if not os.path.isabs(
        config["basic"]["save_file_dir"]) else config["basic"]["save_file_dir"]
    save_file_dir = os.path.abspath(save_file_dir)
    default_task = config["basic"]["default_task"]
    default_website = config["basic"]["default_website"]

    # Experiment settings
    task_file_path = os.path.join(base_dir, config["experiment"]["task_file_path"]) if not os.path.isabs(
        config["experiment"]["task_file_path"]) else config["experiment"]["task_file_path"]
    overwrite = config["experiment"]["overwrite"]
    top_k = config["experiment"]["top_k"]
    fixed_choice_batch_size = config["experiment"]["fixed_choice_batch_size"]
    dynamic_choice_batch_size = config["experiment"]["dynamic_choice_batch_size"]
    max_continuous_no_op = config["experiment"]["max_continuous_no_op"]
    max_op = config["experiment"]["max_op"]
    highlight = config["experiment"]["highlight"]
    monitor = config["experiment"]["monitor"]
    dev_mode = config["experiment"]["dev_mode"]

    try:
        storage_state = config["basic"]["storage_state"]
    except:
        storage_state = None

    # openai settings
    openai_config = config["openai"]
    # Prefer env var, but allow config override if provided
    api_key = openai_config.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "Your API Key Here":
        raise Exception(
            "Missing OpenAI API key. Set OPENAI_API_KEY in your environment or add 'api_key' under [openai] in the config.")

    # playwright settings
    save_video = config["playwright"]["save_video"]
    viewport_size = config["playwright"]["viewport"]
    tracing = config["playwright"]["tracing"]
    locale = None
    try:
        locale = config["playwright"]["locale"]
    except:
        pass
    geolocation = None
    try:
        geolocation = config["playwright"]["geolocation"]
    except:
        pass
    trace_screenshots = config["playwright"]["trace"]["screenshots"]
    trace_snapshots = config["playwright"]["trace"]["snapshots"]
    trace_sources = config["playwright"]["trace"]["sources"]

    # runtime settings (local vs remote CDP like Browserbase)
    runtime = config.get("runtime", {}) or {}
    provider = str(runtime.get("provider", "local")).lower()
    cdp_url = runtime.get("cdp_url")
    headers = runtime.get("headers", {}) or {}
    # Expand env vars in URL and headers
    if isinstance(cdp_url, str):
        cdp_url = os.path.expandvars(cdp_url)
    if isinstance(headers, dict):
        headers = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in headers.items()}

    # Initialize Inference Engine based on OpenAI API
    generation_model = OpenaiEngine(api_key=api_key, **{k: v for k, v in openai_config.items() if k != "api_key"})

    # Load ranking model for prune candidate elements
    ranking_model = None
    if ranker_enabled:
        try:
            from seeact.demo_utils.ranking_model import CrossEncoder
            if torch is None:
                await aprint("Ranking model disabled: 'torch' not installed; proceed without ranker")
                ranker_enabled = False
            else:
                ranking_model = CrossEncoder(
                    ranker_path,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    num_labels=1,
                    max_length=512,
                )
        except Exception as e:
            await aprint(f"Ranking model disabled due to import/init error: {e}")
            ranker_enabled = False

    if not is_demo:
        with open(f'{task_file_path}', 'r', encoding='utf-8') as file:
            query_tasks = json.load(file)
    else:
        query_tasks = []
        task_dict = {}
        task_input = await ainput(
            f"Please input a task, and press Enter. \nOr directly press Enter to use the default task: {default_task}\nTask: ")
        if not task_input:
            task_input = default_task
        task_dict["confirmed_task"] = task_input
        website_input = await ainput(
            f"Please input the complete URL of the starting website, and press Enter. The URL must be complete (for example, including http), to ensure the browser can successfully load the webpage. \nOr directly press Enter to use the default website: {default_website}\nWebsite: ")
        if not website_input:
            website_input = default_website
        task_dict["website"] = website_input
        # set the folder name as current time
        current_time = datetime.datetime.now()
        file_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        task_dict["task_id"] = file_name
        query_tasks.append(task_dict)

    executed_any = False
    for single_query_task in query_tasks:
        confirmed_task = single_query_task["confirmed_task"]
        confirmed_website = single_query_task["website"]
        try:
            confirmed_website_url = website_dict[confirmed_website]
        except:
            confirmed_website_url = confirmed_website
        task_id = single_query_task["task_id"]
        main_result_path = os.path.join(save_file_dir, task_id)

        if not os.path.exists(main_result_path):
            os.makedirs(main_result_path)
            executed_any = True
        else:
            await aprint(f"{main_result_path} already exists")
            if not overwrite:
                continue
        saveconfig(config, os.path.join(main_result_path, "config.toml"))

        # init logger
        # logger = await setup_logger(task_id, main_result_path)
        logger = logging.getLogger(f"{task_id}")
        logger.setLevel(logging.INFO)
        log_fh = logging.FileHandler(os.path.join(main_result_path, f'{task_id}.log'), encoding='utf-8')
        log_fh.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        log_format = logging.Formatter('%(asctime)s - %(message)s')
        terminal_format = logging.Formatter('%(message)s')
        log_fh.setFormatter(log_format)
        console_handler.setFormatter(terminal_format)
        logger.addHandler(log_fh)
        logger.addHandler(console_handler)

        logger.info(f"website: {confirmed_website_url}")
        logger.info(f"task: {confirmed_task}")
        logger.info(f"id: {task_id}")
        
        # Unified interaction loop for all providers (local, cdp, browserbase)
        async def interaction_loop():
            nonlocal ranker_enabled
            taken_actions = []
            complete_flag = False
            monitor_signal = ""
            time_step = 0
            no_op_count = 0
            valid_op_count = 0
            # Repeat/no-progress guard state
            last_url: str | None = None
            last_target_key: str | None = None
            repeat_clicks: int = 0

            while not complete_flag:
                if dev_mode:
                    logger.info(f"Page at the start: {session_control.active_page}")
                await session_control.active_page.bring_to_front()
                terminal_width = 10
                logger.info("=" * terminal_width)
                logger.info(f"Time step: {time_step}")
                logger.info('-' * 10)
                # Heuristic: try dismissing common overlays/popups before scanning
                try:
                    dismissed = await auto_dismiss_overlays(session_control.active_page, max_clicks=2)
                    if dismissed and dev_mode:
                        logger.info(f"Auto-dismissed overlays: {dismissed}")
                except Exception as _e:
                    if dev_mode:
                        logger.info(f"Overlay auto-dismiss error: {_e}")

                # Collect interactive elements. The helper returns dict-shaped entries.
                # Pass viewport to avoid empty results due to None in normalization math.
                # Measure element scan time
                t_scan_start = time.time()
                t_scan_start = time.time()
                try:
                    raw_elements = await get_interactive_elements_with_playwright(
                        session_control.active_page, viewport_size
                    )
                except TypeError:
                    # Backward compatibility if stubbed/mocked without viewport parameter
                    raw_elements = await get_interactive_elements_with_playwright(
                        session_control.active_page
                    )
                t_scan_ms = int((time.time() - t_scan_start) * 1000)
                t_scan_ms = int((time.time() - t_scan_start) * 1000)
                # Adapt to legacy list/tuple structure expected by this CLI loop:
                # [center_point, description, tag_with_role, box_model, selector, real_tag_name]
                if raw_elements and isinstance(raw_elements[0], dict):
                    elements = [
                        [
                            el.get("center_point"),
                            el.get("description"),
                            el.get("tag_with_role"),
                            el.get("box"),
                            el.get("selector"),
                            el.get("tag"),
                        ]
                        for el in raw_elements
                    ]
                else:
                    elements = raw_elements

                # Coverage diagnostics (rough counts)
                try:
                    price_inputs = sum(1 for el in (raw_elements or []) if (
                        (el.get("tag") in ("input", "textarea")) and any(k in (el.get("description") or "").lower() for k in ("price", "min", "max"))
                    ))
                    sliders = sum(1 for el in (raw_elements or []) if 'role="slider"' in (el.get("tag_with_role") or ""))
                    size_controls = sum(1 for el in (raw_elements or []) if (
                        (el.get("tag") in ("button", "label", "a")) and any(s in (el.get("description") or "").lower() for s in ("size", "us 10", "10"))
                    ))
                    sort_controls = sum(1 for el in (raw_elements or []) if (
                        (el.get("tag") == "select") or ("role=\"option\"" in (el.get("tag_with_role") or "")) or ("Options:" in (el.get("description") or ""))
                    ))
                    product_cards = sum(1 for el in (raw_elements or []) if (
                        (el.get("tag") in ("a", "div", "span")) and ("alt=\"" in (el.get("outer_html") or "") or "title=\"" in (el.get("outer_html") or ""))
                    ))
                    logger.info(f"coverage: scan_ms={t_scan_ms} price_inputs={price_inputs} sliders={sliders} size_controls={size_controls} sort_controls={sort_controls} product_cards={product_cards}")
                except Exception:
                    pass

                if tracing:
                    await session_control.context.tracing.start_chunk(title=f'{task_id}-Time Step-{time_step}',
                                                                      name=f"{time_step}")
                logger.info(f"# all elements: {len(elements)}")
                if dev_mode:
                    for i in elements:
                        logger.info(i[1:])
                time_step += 1

                if len(elements) == 0:
                    if monitor:
                        logger.info(
                            f"----------There is no element in this page. Do you want to terminate or continue afterhuman intervention? [i/e].\ni(Intervene): Reject this action, and pause for human intervention.\ne(Exit): Terminate the program and save results.")
                        monitor_input = await ainput()
                        logger.info("Monitor Command: " + monitor_input)
                        if monitor_input in ["i", "intervene", 'intervention']:
                            logger.info(
                                "Pause for human intervention. Press Enter to continue. You can also enter your message here, which will be included in the action history as a human message.")
                            human_intervention = await ainput()
                            if human_intervention:
                                human_intervention = f"Human intervention with a message: {human_intervention}"
                            else:
                                human_intervention = f"Human intervention"
                            taken_actions.append(human_intervention)
                            continue

                    logger.info("Terminate because there is no element in this page.")
                    logger.info("Action History:")
                    for action in taken_actions:
                        logger.info(action)
                    logger.info("")
                    if tracing:
                        logger.info("Save playwright trace file ")
                        await session_control.context.tracing.stop_chunk(
                            path=f"{os.path.join(main_result_path, 'playwright_traces', f'{time_step}.zip')}")

                    logger.info(f"Write results to json file: {os.path.join(main_result_path, 'result.json')}")
                    success_or_not = ""
                    if valid_op_count == 0:
                        success_or_not = "0"
                    final_json = {"confirmed_task": confirmed_task, "website": confirmed_website,
                                  "task_id": task_id, "success_or_not": success_or_not,
                                  "num_step": len(taken_actions), "action_history": taken_actions,
                                  "exit_by": "No elements"}

                    with open(os.path.join(main_result_path, 'result.json'), 'w', encoding='utf-8') as file:
                        json.dump(final_json, file, indent=4)
                    logger.info("Close browser context")
                    logger.removeHandler(log_fh)
                    logger.removeHandler(console_handler)

                    close_context = session_control.context
                    session_control.context = None
                    await close_context.close()
                    return
                # Include DOM snippet in choice text if enabled
                include_dom_in_choices = False
                try:
                    include_dom_in_choices = bool(config.get("experiment", {}).get("include_dom_in_choices", False))
                except Exception:
                    include_dom_in_choices = False

                if ranker_enabled and len(elements) > top_k:
                    ranking_input = format_ranking_input(elements, confirmed_task, taken_actions)
                    logger.info("Start to rank")
                    try:
                        from seeact.demo_utils.ranking_model import find_topk
                        pred_scores = ranking_model.predict(
                            ranking_input,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                            batch_size=100,
                        )
                        topk_values, topk_indices = find_topk(
                            pred_scores, k=min(top_k, len(elements))
                        )
                    except Exception as e:
                        logger.info(f"Ranking failed; falling back to all elements: {e}")
                        ranker_enabled = False
                        all_candidate_ids = range(len(elements))
                        ranked_elements = elements
                    else:
                        all_candidate_ids = list(topk_indices)
                        ranked_elements = [elements[i] for i in all_candidate_ids]
                else:

                    all_candidate_ids = range(len(elements))
                    ranked_elements = elements

                all_candidate_ids_with_location = []
                for element_id, element_detail in zip(all_candidate_ids, ranked_elements):
                    all_candidate_ids_with_location.append(
                        (element_id, round(element_detail[0][1]), round(element_detail[0][0])))

                all_candidate_ids_with_location.sort(key=lambda x: (x[1], x[2]))

                all_candidate_ids = [element_id[0] for element_id in all_candidate_ids_with_location]
                num_choices = len(all_candidate_ids)
                if ranker_enabled:
                    logger.info(f"# element candidates: {num_choices}")

                total_height = await session_control.active_page.evaluate('''() => {
                                                                return Math.max(
                                                                    document.documentElement.scrollHeight, 
                                                                    document.body.scrollHeight,
                                                                    document.documentElement.clientHeight
                                                                );
                                                            }''')
                if dynamic_choice_batch_size > 0:
                    step_length = min(num_choices,
                                      num_choices // max(round(total_height / dynamic_choice_batch_size), 1) + 1)
                else:
                    step_length = min(num_choices, fixed_choice_batch_size)
                logger.info(f"batch size: {step_length}")
                logger.info('-' * 10)

                _vw = session_control.active_page.viewport_size or viewport_size
                total_width = _vw["width"]
                log_task = "You are asked to complete the following task: " + confirmed_task
                logger.info(log_task)
                previous_actions = taken_actions

                previous_action_text = "Previous Actions:\n"
                if previous_actions is None or previous_actions == []:
                    previous_action_text += "None"
                else:
                    for i in previous_actions:
                        previous_action_text += i + '\n'
                logger.info(previous_action_text)
                logger.info('-' * 10)
                logger.info("Start Multi-Choice QA - Batch 0")
                # Initialize per-step action targets to avoid UnboundLocalError
                new_action = ""
                got_one_answer = False
                query_count = 0
                target_element = []
                target_element_text = ""
                target_action = ""
                target_value = ""

                for multichoice_i in range(0, len(all_candidate_ids), step_length):
                    if got_one_answer:
                        break

                    input_image_path = os.path.join(main_result_path, 'image_inputs',
                                                    f'{time_step}_{multichoice_i // step_length}_crop.jpg')

                    height_start = all_candidate_ids_with_location[multichoice_i][1]
                    height_end = all_candidate_ids_with_location[min(multichoice_i + step_length, num_choices) - 1][1]

                    total_height = await session_control.active_page.evaluate('''() => {
                                                                    return Math.max(
                                                                        document.documentElement.scrollHeight, 
                                                                        document.body.scrollHeight,
                                                                        document.documentElement.clientHeight
                                                                    );
                                                                }''')
                    clip_start = min(total_height - 1144, max(0, height_start - 200))
                    clip_height = min(total_height - clip_start, max(height_end - height_start + 200, 1144))
                    clip = {"x": 0, "y": clip_start, "width": total_width, "height": clip_height}

                    if dev_mode:
                        logger.info(height_start)
                        logger.info(height_end)
                        logger.info(total_height)
                        logger.info(clip)

                    # Capture a compact JPEG (quality ~75) for speed; avoid full_page when clipping
                    t_ss_start = time.time()
                    try:
                        await session_control.active_page.screenshot(
                            path=input_image_path,
                            clip=clip,
                            type='jpeg',
                            quality=75,
                            timeout=10000
                        )
                    except Exception as e_clip:
                        logger.info(f"Failed to get cropped screenshot because {e_clip}")
                    ss_ms = int((time.time() - t_ss_start) * 1000)
                    try:
                        img_bytes = os.path.getsize(input_image_path) if os.path.exists(input_image_path) else 0
                    except Exception:
                        img_bytes = 0

                    if dev_mode:
                        logger.info(multichoice_i)
                    if not os.path.exists(input_image_path):
                        if dev_mode:
                            logger.info("No screenshot")
                        continue
                    candidate_ids = all_candidate_ids[multichoice_i:multichoice_i + step_length]
                    try:
                        batch_dicts = [raw_elements[i] for i in candidate_ids] if raw_elements else []
                    except Exception:
                        batch_dicts = []
                    choices = format_choices(batch_dicts, include_dom=include_dom_in_choices)
                    query_count += 1
                    # Format prompts for LLM inference
                    prompt = generate_prompt(task=confirmed_task, previous=taken_actions, choices=choices,
                                             experiment_split="SeeAct")
                    if dev_mode:
                        for prompt_i in prompt:
                            logger.info("%s", prompt_i)

                    # One-turn decision: call only once with the referring prompt; log a stub for the planning section
                    terminal_width = 10
                    logger.info("-" * terminal_width)
                    logger.info("🤖Action Generation Output🤖")
                    logger.info("One-turn mode: generating structured decision in a single call.")
                    logger.info("-" * (terminal_width))

                    choice_text = f"(Multichoice Question) - Batch {multichoice_i // step_length}" + "\n" + format_options(
                        choices)
                    choice_text = choice_text.replace("\n\n", "")

                    for line in choice_text.split('\n'):
                        logger.info("%s", line)

                    t_llm_start = time.time()
                    try:
                        output = generation_model.generate(
                            prompt=prompt,
                            image_path=input_image_path,
                            turn_number=1,
                            ouput_0="",
                            max_new_tokens=384,
                            image_detail="auto",
                        )
                    except Exception as e:
                        logger.info("Model generation error (grounding): %s", e)
                        none_letter = generate_option_name(len(choices)) if choices else "A"
                        output = f"ELEMENT: {none_letter}\nACTION: NONE\nVALUE: None"
                    llm_ms = int((time.time() - t_llm_start) * 1000)

                    terminal_width = 10
                    logger.info("-" * terminal_width)
                    logger.info("🤖Grounding Output🤖")

                    for line in output.split('\n'):
                        logger.info("%s", line)
                    pred_element, pred_action, pred_value = postprocess_action_lmm(output)
                    if len(pred_element) in [1, 2]:
                        element_id = get_index_from_option_name(pred_element)
                    else:
                        element_id = -1

                    # Process the elements
                    if (0 <= element_id < len(candidate_ids) and pred_action.strip() in ["CLICK", "SELECT", "TYPE",
                                                                                         "PRESS ENTER", "HOVER",
                                                                                         "TERMINATE"]):
                        # Map chosen option (0..len(candidate_ids)-1) back to original element index
                        target_element = elements[candidate_ids[element_id]]
                        # choices is a list of textual strings; use the selected one directly
                        target_element_text = choices[element_id]
                        target_action = pred_action
                        target_value = pred_value
                        new_action += "[" + target_element[2] + "]" + " "
                        new_action += target_element[1] + " -> " + target_action
                        if target_action.strip() in ["SELECT", "TYPE"]:
                            new_action += ": " + target_value
                        got_one_answer = True
                        break
                    elif pred_action.strip() in ["PRESS ENTER", "TERMINATE"]:
                        target_element = pred_action
                        target_element_text = target_element
                        target_action = pred_action
                        target_value = pred_value
                        new_action += target_action
                        if target_action.strip() in ["SELECT", "TYPE"]:
                            new_action += ": " + target_value
                        got_one_answer = True
                        break
                    else:
                        # Fuzzy fallback
                        def _extract_plan_targets(text: str):
                            import re as _re
                            cands = []
                            cands += _re.findall(r'"([^"]{2,80})"', text)
                            cands += _re.findall(r"'([^']{2,80})'", text)
                            verb = _re.search(r"\b(click|select|type|press|goto|go to)\b\s+([^\n\.;:,]{2,80})", text, _re.I)
                            if verb:
                                cands.append(verb.group(2))
                            return [s.strip() for s in cands if s.strip()]

                        def _parse_action_from_plan(text: str):
                            tl = text.lower()
                            if "press enter" in tl:
                                return "PRESS ENTER", ""
                            if "type" in tl:
                                import re as _re
                                m = _re.search(r"type\s+\"([^\"]+)\"", text, _re.I)
                                return "TYPE", (m.group(1) if m else "")
                            if "select" in tl:
                                return "SELECT", ""
                            if "click" in tl:
                                return "CLICK", ""
                            return "CLICK", ""

                        def _clean_choice(s: str) -> str:
                            base = s.split(" | DOM:", 1)[0]
                            if ") " in base:
                                base = base.split(") ", 1)[1]
                            import re as _re
                            base = _re.sub(r"<[^>]+>", " ", base)
                            return base.strip().lower()

                        plan_targets = _extract_plan_targets(output)
                        if plan_targets and choices:
                            best = (-1.0, None)
                            for idx, txt in enumerate(choices):
                                c = _clean_choice(txt)
                                for t in plan_targets:
                                    score = SequenceMatcher(None, c, t.lower()).ratio()
                                    if score > best[0]:
                                        best = (score, idx)
                            if best[1] is not None and best[0] >= 0.45:
                                fb_idx = best[1]
                                target_element = elements[candidate_ids[fb_idx]]
                                target_element_text = choices[fb_idx]
                                action_guess, value_guess = _parse_action_from_plan(output)
                                target_action = action_guess
                                target_value = value_guess
                                new_action += "[" + target_element[2] + "]" + " "
                                new_action += target_element[1] + " -> " + target_action
                                if target_action in ["SELECT", "TYPE"] and target_value:
                                    new_action += ": " + target_value
                                got_one_answer = True
                                break

                    # Log per-batch metrics
                    try:
                        prompt_text = "\n".join(str(p) for p in (prompt or []))
                        approx_tokens = int(len(prompt_text) / 4)
                        logger.info(f"metrics: scan_ms={t_scan_ms} ss_ms={ss_ms} img_bytes={img_bytes} llm_ms={llm_ms} approx_tokens={approx_tokens}")
                    except Exception:
                        pass

                if got_one_answer:
                    terminal_width = 10
                    logger.info("-" * terminal_width)
                    logger.info("🤖Browser Operation🤖")
                    logger.info(f"Target Element: {target_element_text}", )
                    logger.info(f"Target Action: {target_action}", )
                    logger.info(f"Target Value: {target_value}", )

                    # Repeat/no-progress guard: if same target on same URL repeatedly, suppress and nudge scroll
                    try:
                        _url_now = session_control.active_page.url
                    except Exception:
                        _url_now = ""
                    _target_key = f"{target_action}|{target_element_text}".strip()
                    if _url_now == (last_url or "") and _target_key and _target_key == (last_target_key or ""):
                        repeat_clicks += 1
                    else:
                        repeat_clicks = 0

                    if repeat_clicks >= 1 and target_action == "CLICK":
                        try:
                            await session_control.active_page.evaluate(
                                "window.scrollBy(0, Math.min(window.innerHeight * 0.6, 600));"
                            )
                        except Exception:
                            pass
                        taken_actions.append(
                            f"Auto-nudge: suppressed repeat of {target_action} {target_element_text}; scrolled"
                        )
                        if monitor_signal not in ["pause", "reject"]:
                            no_op_count += 1
                        time_step += 1
                        last_url = _url_now
                        last_target_key = _target_key
                        continue

                    # Repeat/no-progress guard: if same target on same URL repeatedly, suppress and nudge scroll
                    try:
                        _url_now = session_control.active_page.url
                    except Exception:
                        _url_now = ""
                    _target_key = f"{target_action}|{target_element_text}".strip()
                    if _url_now == (last_url or "") and _target_key and _target_key == (last_target_key or ""):
                        repeat_clicks += 1
                    else:
                        repeat_clicks = 0

                    if repeat_clicks >= 1 and target_action == "CLICK":
                        try:
                            await session_control.active_page.evaluate(
                                "window.scrollBy(0, Math.min(window.innerHeight * 0.6, 600));"
                            )
                        except Exception:
                            pass
                        taken_actions.append(
                            f"Auto-nudge: suppressed repeat of {target_action} {target_element_text}; scrolled"
                        )
                        if monitor_signal not in ["pause", "reject"]:
                            no_op_count += 1
                        time_step += 1
                        # Skip executing this repeated action
                        last_url = _url_now
                        last_target_key = _target_key
                        continue

                    if monitor:
                        logger.info(
                            f"----------\nShould I execute the above action? [Y/n/i/e].\nY/n: Accept or reject this action.\ni(Intervene): Reject this action, and pause for human intervention.\ne(Exit): Terminate the program and save results.")
                        monitor_input = await ainput()
                        logger.info("Monitor Command: " + monitor_input)
                        if monitor_input in ["n", "N", "No", "no"]:
                            monitor_signal = "reject"
                            target_element = []
                        elif monitor_input in ["e", "exit", "Exit"]:
                            monitor_signal = "exit"
                        elif monitor_input in ["i", "intervene", 'intervention']:
                            monitor_signal = "pause"
                            target_element = []
                        else:
                            valid_op_count += 1
                else:
                    no_op_count += 1
                    target_element = []

                try:
                    if monitor_signal == 'exit':
                        raise Exception("human supervisor manually made it exit.")
                    if no_op_count >= max_continuous_no_op:
                        raise Exception(f"no executable operations for {max_continuous_no_op} times.")
                    elif time_step >= max_op:
                        raise Exception(f"the agent reached the step limit {max_op}.")
                    elif got_one_answer and target_action == "TERMINATE":
                        raise Exception("The model determined a completion.")

                    selector = None
                    fail_to_execute = False
                    try:
                        if target_element == []:
                            pass
                        else:
                            if not target_element in ["PRESS ENTER", "TERMINATE"]:
                                selector = target_element[-2]
                                if dev_mode:
                                    logger.info(target_element)
                                try:
                                    await selector.scroll_into_view_if_needed(timeout=3000)
                                    if highlight:
                                        await selector.highlight()
                                        await asyncio.sleep(2.5)
                                except Exception as e:
                                    pass

                        if selector:
                            valid_op_count += 1
                            if target_action == "CLICK":
                                js_click = True
                                try:
                                    if target_element[-1] in ["select", "input"]:
                                        logger.info("Try performing a CLICK")
                                        await selector.evaluate("element => element.click()", timeout=10000)
                                        js_click = False
                                    else:
                                        await selector.click(timeout=10000)
                                except Exception as e:
                                    try:
                                        if not js_click:
                                            logger.info("Try performing a CLICK")
                                            await selector.evaluate("element => element.click()", timeout=10000)
                                        else:
                                            raise Exception(e)
                                    except Exception as ee:
                                        try:
                                            logger.info("Try performing a HOVER")
                                            await selector.hover(timeout=10000)
                                            new_action = new_action.replace("CLICK",
                                                                            f"Failed to CLICK because {e}, did a HOVER instead")
                                        except Exception as eee:
                                            new_action = new_action.replace("CLICK", f"Failed to CLICK because {e}")
                                            no_op_count += 1
                            elif target_action == "TYPE":
                                try:
                                    try:
                                        logger.info("Try performing a \"press_sequentially\"")
                                        await selector.clear(timeout=10000)
                                        await selector.fill("", timeout=10000)
                                        await selector.press_sequentially(target_value, timeout=10000)
                                    except Exception as e0:
                                        await selector.fill(target_value, timeout=10000)
                                except Exception as e:
                                    try:
                                        if target_element[-1] in ["select"]:
                                            logger.info("Try performing a SELECT")
                                            selected_value = await select_option(selector, target_value)
                                            new_action = new_action.replace("TYPE",
                                                                            f"Failed to TYPE \"{target_value}\" because {e}, did a SELECT {selected_value} instead")
                                        else:
                                            raise Exception(e)
                                    except Exception as ee:
                                        js_click = True
                                        try:
                                            if target_element[-1] in ["select", "input"]:
                                                logger.info("Try performing a CLICK")
                                                await selector.evaluate("element => element.click()", timeout=10000)
                                                js_click = False
                                            else:
                                                logger.info("Try performing a CLICK")
                                                await selector.click(timeout=10000)
                                            new_action = "[" + target_element[2] + "]" + " "
                                            new_action += target_element[
                                                              1] + " -> " + f"Failed to TYPE \"{target_value}\" because {e}, did a CLICK instead"
                                        except Exception as eee:
                                            try:
                                                if not js_click:
                                                    if dev_mode:
                                                        logger.info(eee)
                                                    logger.info("Try performing a CLICK")
                                                    await selector.evaluate("element => element.click()", timeout=10000)
                                                    new_action = "[" + target_element[2] + "]" + " "
                                                    new_action += target_element[
                                                                      1] + " -> " + f"Failed to TYPE \"{target_value}\" because {e}, did a CLICK instead"
                                                else:
                                                    raise Exception(eee)
                                            except Exception as eeee:
                                                new_action = "[" + target_element[2] + "]" + " "
                                                new_action += target_element[
                                                                  1] + " -> " + f"Failed to HOVER because {e}"
                                                no_op_count += 1
                            elif target_action == "PRESS ENTER":
                                try:
                                    logger.info("Try performing a PRESS ENTER")
                                    await selector.press('Enter')
                                except Exception as e:
                                    await selector.click(timeout=10000)
                                    await session_control.active_page.keyboard.press('Enter')
                        elif monitor_signal == "pause":
                            logger.info(
                                "Pause for human intervention. Press Enter to continue. You can also enter your message here, which will be included in the action history as a human message.")
                            human_intervention = await ainput()
                            if human_intervention:
                                human_intervention = f" Human message: {human_intervention}"
                            raise Exception(
                                f"the human supervisor rejected this operation and may have taken some actions.{human_intervention}")
                        elif monitor_signal == "reject":
                            raise Exception("the human supervisor rejected this operation.")
                        elif target_element == "PRESS ENTER":
                            logger.info("Try performing a PRESS ENTER")
                            await session_control.active_page.keyboard.press('Enter')
                        no_op_count = 0
                        try:
                            await session_control.active_page.wait_for_load_state('load')
                        except Exception as e:
                            pass
                    except Exception as e:
                        if target_action not in ["TYPE", "SELECT"]:
                            new_action = f"Failed to {target_action} {target_element_text} because {e}"

                        else:
                            new_action = f"Failed to {target_action} {target_value} for {target_element_text} because {e}"
                        fail_to_execute = True

                    if new_action == "" or fail_to_execute:
                        if new_action == "":
                            new_action = "No Operation"
                        if monitor_signal not in ["pause", "reject"]:
                            no_op_count += 1
                    taken_actions.append(new_action)
                    # Update repeat/no-progress guard state after execution attempt
                    try:
                        last_url = session_control.active_page.url
                    except Exception:
                        pass
                    last_target_key = _target_key if got_one_answer else last_target_key
                    # Update repeat/no-progress guard state after execution attempt
                    try:
                        last_url = session_control.active_page.url
                    except Exception:
                        pass
                    last_target_key = _target_key if got_one_answer else last_target_key
                    if not session_control.context.pages:
                        await session_control.context.new_page()
                        try:
                            await session_control.active_page.goto(confirmed_website_url, wait_until="load")
                        except Exception as e:
                            pass

                    if monitor_signal == 'pause':
                        pass
                    else:
                        await asyncio.sleep(0.3)
                    if dev_mode:
                        logger.info(f"current active page: {session_control.active_page}")
                        logger.info("All pages")
                        logger.info(session_control.context.pages)
                        logger.info("-" * 10)
                    try:
                        await session_control.active_page.wait_for_load_state('load')
                    except Exception as e:
                        if dev_mode:
                            logger.info(e)
                    if tracing:
                        logger.info("Save playwright trace file")
                        await session_control.context.tracing.stop_chunk(
                            path=f"{os.path.join(main_result_path, 'playwright_traces', f'{time_step}.zip')}")
                except Exception as e:
                    logger.info("=" * 10)
                    logger.info(f"Decide to terminate because {e}")
                    logger.info("Action History:")

                    for action in taken_actions:
                        logger.info(action)
                    logger.info("")

                    if tracing:
                        logger.info("Save playwright trace file")
                        await session_control.context.tracing.stop_chunk(
                            path=f"{os.path.join(main_result_path, 'playwright_traces', f'{time_step}.zip')}")

                    success_or_not = ""
                    if valid_op_count == 0:
                        success_or_not = "0"
                    logger.info(f"Write results to json file: {os.path.join(main_result_path, 'result.json')}")
                    final_json = {"confirmed_task": confirmed_task, "website": confirmed_website,
                                  "task_id": task_id, "success_or_not": success_or_not,
                                  "num_step": len(taken_actions), "action_history": taken_actions, "exit_by": str(e)}

                    with open(os.path.join(main_result_path, 'result.json'), 'w', encoding='utf-8') as file:
                        json.dump(final_json, file, indent=4)

                    if monitor:
                        logger.info("Wait for human inspection. Directly press Enter to exit")
                        monitor_input = await ainput()

                    logger.info("Close browser context")
                    logger.removeHandler(log_fh)
                    logger.removeHandler(console_handler)
                    close_context = session_control.context
                    session_control.context = None
                    await close_context.close()
                    return
    # If nothing to run (e.g., overwrite=false and all outputs exist), exit gracefully
    if not executed_any and not is_demo:
        await aprint("No tasks to run: all outputs exist and overwrite=false. Use the runner for batch runs or set overwrite=true.")
        return

    async with async_playwright() as playwright:
        # Choose local browser or connect over CDP (e.g., Browserbase)
        if provider in ("cdp",) and cdp_url:
            session_control.browser = await playwright.chromium.connect_over_cdp(cdp_url, headers=headers)
            # Prefer an existing context for remote connections
            if getattr(session_control.browser, "contexts", None):
                if session_control.browser.contexts:
                    session_control.context = session_control.browser.contexts[0]
            if session_control.context is None:
                session_control.context = await normal_new_context_async(session_control.browser,
                                                                         tracing=tracing,
                                                                         storage_state=storage_state,
                                                                         video_path=main_result_path if save_video else None,
                                                                         viewport=viewport_size,
                                                                         trace_screenshots=trace_screenshots,
                                                                         trace_snapshots=trace_snapshots,
                                                                         trace_sources=trace_sources,
                                                                         geolocation=geolocation,
                                                                         locale=locale)
            # Run unified loop for CDP provider
            session_control.context.on("page", page_on_open_handler)
            pages = session_control.context.pages
            if pages:
                page = pages[-1]
            else:
                page = await session_control.context.new_page()
            try:
                await page_on_open_handler(page)
            except Exception:
                pass
            try:
                # Ensure viewport is set when connecting over CDP (can be None by default)
                try:
                    if not page.viewport_size:
                        await page.set_viewport_size(viewport_size)
                except Exception:
                    pass
                await page.goto(confirmed_website_url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:
                logger.info("Failed to fully load the webpage before timeout")
                logger.info(e)
            await asyncio.sleep(1)
            await interaction_loop()
        elif provider in ("browserbase",):
            if bb_create is None:
                raise RuntimeError("Browserbase runtime requested but client not available. Ensure src/runtime/browserbase_client.py is present and dependencies installed.")
            pid = runtime.get("project_id") or os.getenv("BROWSERBASE_PROJECT_ID")
            key = runtime.get("api_key") or os.getenv("BROWSERBASE_API_KEY")
            api_base = runtime.get("api_base") or os.getenv("BROWSERBASE_API_BASE")
            project_id, api_key = bb_resolve(pid, key)  # type: ignore
            ws_url, session_id = bb_create(project_id, api_key, api_base=api_base)  # type: ignore
            bb_session_id = session_id
            session_control.browser = await playwright.chromium.connect_over_cdp(ws_url)
            if getattr(session_control.browser, "contexts", None):
                if session_control.browser.contexts:
                    session_control.context = session_control.browser.contexts[0]
            if session_control.context is None:
                session_control.context = await normal_new_context_async(session_control.browser,
                                                                         tracing=tracing,
                                                                         storage_state=storage_state,
                                                                         video_path=main_result_path if save_video else None,
                                                                         viewport=viewport_size,
                                                                         trace_screenshots=trace_screenshots,
                                                                         trace_snapshots=trace_snapshots,
                                                                         trace_sources=trace_sources,
                                                                         geolocation=geolocation,
                                                                         locale=locale)
            # Run unified loop for Browserbase provider
            session_control.context.on("page", page_on_open_handler)
            pages = session_control.context.pages
            if pages:
                page = pages[-1]
            else:
                page = await session_control.context.new_page()
            try:
                await page_on_open_handler(page)
            except Exception:
                pass
            try:
                try:
                    if not page.viewport_size:
                        await page.set_viewport_size(viewport_size)
                except Exception:
                    pass
                await page.goto(confirmed_website_url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:
                logger.info("Failed to fully load the webpage before timeout")
                logger.info(e)
            await asyncio.sleep(1)
            await interaction_loop()
            try:
                if bb_close is not None and bb_session_id:
                    bb_close(bb_session_id)
            except Exception as _e:
                if dev_mode:
                    logger.info(f"Failed to close Browserbase session: {_e}")
        else:
            session_control.browser = await normal_launch_async(playwright)
            session_control.context = await normal_new_context_async(session_control.browser,
                                                                     tracing=tracing,
                                                                     storage_state=storage_state,
                                                                     video_path=main_result_path if save_video else None,
                                                                     viewport=viewport_size,
                                                                     trace_screenshots=trace_screenshots,
                                                                     trace_snapshots=trace_snapshots,
                                                                     trace_sources=trace_sources,
                                                                     geolocation=geolocation,
                                                                     locale=locale)
            session_control.context.on("page", page_on_open_handler)
            page = await session_control.context.new_page()
            try:
                await page_on_open_handler(page)
            except Exception:
                pass
            try:
                try:
                    if not page.viewport_size:
                        await page.set_viewport_size(viewport_size)
                except Exception:
                    pass
                await page.goto(confirmed_website_url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:
                logger.info("Failed to fully load the webpage before timeout")
                logger.info(e)
            await asyncio.sleep(1)

            taken_actions = []
            complete_flag = False
            monitor_signal = ""
            time_step = 0
            no_op_count = 0
            valid_op_count = 0

            while not complete_flag:
                if dev_mode:
                    logger.info(f"Page at the start: {session_control.active_page}")
                await session_control.active_page.bring_to_front()
                terminal_width = 10
                logger.info("=" * terminal_width)
                logger.info(f"Time step: {time_step}")
                logger.info('-' * 10)
                # Heuristic: try dismissing common overlays/popups before scanning
                try:
                    dismissed = await auto_dismiss_overlays(session_control.active_page, max_clicks=2)
                    if dismissed and dev_mode:
                        logger.info(f"Auto-dismissed overlays: {dismissed}")
                except Exception as _e:
                    if dev_mode:
                        logger.info(f"Overlay auto-dismiss error: {_e}")
                # Collect interactive elements. The helper returns dict-shaped entries.
                # Pass viewport to avoid empty results due to None in normalization math.
                try:
                    raw_elements = await get_interactive_elements_with_playwright(
                        session_control.active_page, viewport_size
                    )
                except TypeError:
                    # Backward compatibility if stubbed/mocked without viewport parameter
                    raw_elements = await get_interactive_elements_with_playwright(
                        session_control.active_page
                    )
                # Adapt to legacy list/tuple structure expected by this CLI loop:
                # [center_point, description, tag_with_role, box_model, selector, real_tag_name]
                if raw_elements and isinstance(raw_elements[0], dict):
                    elements = [
                        [
                            el.get("center_point"),
                            el.get("description"),
                            el.get("tag_with_role"),
                            el.get("box"),
                            el.get("selector"),
                            el.get("tag"),
                        ]
                        for el in raw_elements
                    ]
                else:
                    elements = raw_elements

                if tracing:
                    await session_control.context.tracing.start_chunk(title=f'{task_id}-Time Step-{time_step}',
                                                                      name=f"{time_step}")
                logger.info(f"# all elements: {len(elements)}")
                if dev_mode:
                    for i in elements:
                        logger.info(i[1:])
                time_step += 1

                if len(elements) == 0:
                    if monitor:
                        logger.info(
                            f"----------There is no element in this page. Do you want to terminate or continue after"
                            f"human intervention? [i/e].\ni(Intervene): Reject this action, and pause for human "
                            f"intervention.\ne(Exit): Terminate the program and save results.")
                        monitor_input = await ainput()
                        logger.info("Monitor Command: " + monitor_input)
                        if monitor_input in ["i", "intervene", 'intervention']:
                            logger.info(
                                "Pause for human intervention. Press Enter to continue. You can also enter your message here, which will be included in the action history as a human message.")
                            human_intervention = await ainput()
                            if human_intervention:
                                human_intervention = f"Human intervention with a message: {human_intervention}"
                            else:
                                human_intervention = f"Human intervention"
                            taken_actions.append(human_intervention)
                            continue

                    logger.info("Terminate because there is no element in this page.")
                    logger.info("Action History:")
                    for action in taken_actions:
                        logger.info(action)
                    logger.info("")
                    if tracing:
                        logger.info("Save playwright trace file ")
                        await session_control.context.tracing.stop_chunk(
                            path=f"{os.path.join(main_result_path, 'playwright_traces', f'{time_step}.zip')}")

                    logger.info(f"Write results to json file: {os.path.join(main_result_path, 'result.json')}")
                    success_or_not = ""
                    if valid_op_count == 0:
                        success_or_not = "0"
                    final_json = {"confirmed_task": confirmed_task, "website": confirmed_website,
                                  "task_id": task_id, "success_or_not": success_or_not,
                                  "num_step": len(taken_actions), "action_history": taken_actions,
                                  "exit_by": "No elements"}

                    with open(os.path.join(main_result_path, 'result.json'), 'w', encoding='utf-8') as file:
                        json.dump(final_json, file, indent=4)
                    # logger.shutdown()
                    #
                    # if monitor:
                    #     logger.info("Wait for human inspection. Directly press Enter to exit")
                    #     monitor_input = await ainput()
                    logger.info("Close browser context")
                    logger.removeHandler(log_fh)
                    logger.removeHandler(console_handler)

                    close_context = session_control.context
                    session_control.context = None
                    await close_context.close()
                    complete_flag = True
                    continue
                # Include DOM snippet in choice text if enabled
                include_dom_in_choices = False
                try:
                    include_dom_in_choices = bool(config.get("experiment", {}).get("include_dom_in_choices", False))
                except Exception:
                    include_dom_in_choices = False

                if ranker_enabled and len(elements) > top_k:
                    ranking_input = format_ranking_input(elements, confirmed_task, taken_actions)
                    logger.info("Start to rank")
                    try:
                        from seeact.demo_utils.ranking_model import find_topk
                        pred_scores = ranking_model.predict(
                            ranking_input,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                            batch_size=100,
                        )
                        topk_values, topk_indices = find_topk(
                            pred_scores, k=min(top_k, len(elements))
                        )
                    except Exception as e:
                        logger.info(f"Ranking failed; falling back to all elements: {e}")
                        ranker_enabled = False
                        all_candidate_ids = range(len(elements))
                        ranked_elements = elements
                    else:
                        all_candidate_ids = list(topk_indices)
                        ranked_elements = [elements[i] for i in all_candidate_ids]
                else:

                    all_candidate_ids = range(len(elements))
                    ranked_elements = elements

                all_candidate_ids_with_location = []
                for element_id, element_detail in zip(all_candidate_ids, ranked_elements):
                    all_candidate_ids_with_location.append(
                        (element_id, round(element_detail[0][1]), round(element_detail[0][0])))

                all_candidate_ids_with_location.sort(key=lambda x: (x[1], x[2]))

                all_candidate_ids = [element_id[0] for element_id in all_candidate_ids_with_location]
                num_choices = len(all_candidate_ids)
                if ranker_enabled:
                    logger.info(f"# element candidates: {num_choices}")

                total_height = await session_control.active_page.evaluate('''() => {
                                                                return Math.max(
                                                                    document.documentElement.scrollHeight, 
                                                                    document.body.scrollHeight,
                                                                    document.documentElement.clientHeight
                                                                );
                                                            }''')
                if dynamic_choice_batch_size > 0:
                    step_length = min(num_choices,
                                      num_choices // max(round(total_height / dynamic_choice_batch_size), 1) + 1)
                else:
                    step_length = min(num_choices, fixed_choice_batch_size)
                logger.info(f"batch size: {step_length}")
                logger.info('-' * 10)

                _vw = session_control.active_page.viewport_size or viewport_size
                total_width = _vw["width"]
                log_task = "You are asked to complete the following task: " + confirmed_task
                logger.info(log_task)
                previous_actions = taken_actions

                previous_action_text = "Previous Actions:\n"
                if previous_actions is None or previous_actions == []:
                    previous_actions = ["None"]
                for action_text in previous_actions:
                    previous_action_text += action_text
                    previous_action_text += "\n"

                log_previous_actions = previous_action_text
                logger.info(log_previous_actions[:-1])

                target_element = []

                new_action = ""
                target_action = "CLICK"
                target_value = ""
                query_count = 0
                got_one_answer = False

                for multichoice_i in range(0, num_choices, step_length):
                    logger.info("-" * 10)
                    logger.info(f"Start Multi-Choice QA - Batch {multichoice_i // step_length}")
                    input_image_path = os.path.join(main_result_path, 'image_inputs',
                                                    f'{time_step}_{multichoice_i // step_length}_crop.jpg')

                    height_start = all_candidate_ids_with_location[multichoice_i][1]
                    height_end = all_candidate_ids_with_location[min(multichoice_i + step_length, num_choices) - 1][1]

                    total_height = await session_control.active_page.evaluate('''() => {
                                                                    return Math.max(
                                                                        document.documentElement.scrollHeight, 
                                                                        document.body.scrollHeight,
                                                                        document.documentElement.clientHeight
                                                                    );
                                                                }''')
                    clip_start = min(total_height - 1144, max(0, height_start - 200))
                    clip_height = min(total_height - clip_start, max(height_end - height_start + 200, 1144))
                    clip = {"x": 0, "y": clip_start, "width": total_width, "height": clip_height}

                    if dev_mode:
                        logger.info(height_start)
                        logger.info(height_end)
                        logger.info(total_height)
                        logger.info(clip)

                    t_ss_start = time.time()
                    try:
                        await session_control.active_page.screenshot(
                            path=input_image_path,
                            clip=clip,
                            type='jpeg',
                            quality=75,
                            timeout=10000
                        )
                    except Exception as e_clip:
                        logger.info(f"Failed to get cropped screenshot because {e_clip}")
                    ss_ms = int((time.time() - t_ss_start) * 1000)
                    try:
                        img_bytes = os.path.getsize(input_image_path) if os.path.exists(input_image_path) else 0
                    except Exception:
                        img_bytes = 0

                    if dev_mode:
                        logger.info(multichoice_i)
                    if not os.path.exists(input_image_path):
                        if dev_mode:
                            logger.info("No screenshot")
                        continue
                    candidate_ids = all_candidate_ids[multichoice_i:multichoice_i + step_length]
                    # Build textual choices for the current batch using dict-shaped originals for formatting
                    try:
                        # Recreate dict-shaped elements for formatter when we adapted to legacy list above
                        # raw_elements aligns 1:1 with elements (same ordering)
                        batch_dicts = [raw_elements[i] for i in candidate_ids] if raw_elements else []
                    except Exception:
                        batch_dicts = []
                    choices = format_choices(batch_dicts, include_dom=include_dom_in_choices)
                    query_count += 1
                    # Format prompts for LLM inference
                    prompt = generate_prompt(task=confirmed_task, previous=taken_actions, choices=choices,
                                             experiment_split="SeeAct")
                    if dev_mode:
                        for prompt_i in prompt:
                            logger.info("%s", prompt_i)

                    terminal_width = 10
                    logger.info("-" * terminal_width)
                    logger.info("🤖Action Generation Output🤖")
                    logger.info("One-turn mode: generating structured decision in a single call.")
                    logger.info("-" * (terminal_width))

                    choice_text = f"(Multichoice Question) - Batch {multichoice_i // step_length}" + "\n" + format_options(
                        choices)
                    choice_text = choice_text.replace("\n\n", "")

                    for line in choice_text.split('\n'):
                        logger.info("%s", line)
                    # logger.info(choice_text)

                    t_llm_start = time.time()
                    try:
                        output = generation_model.generate(
                            prompt=prompt,
                            image_path=input_image_path,
                            turn_number=1,
                            ouput_0="",
                            max_new_tokens=384,
                            image_detail="auto",
                        )
                    except Exception as e:
                        logger.info("Model generation error (grounding): %s", e)
                        none_letter = generate_option_name(len(choices)) if choices else "A"
                        output = f"ELEMENT: {none_letter}\nACTION: NONE\nVALUE: None"
                    llm_ms = int((time.time() - t_llm_start) * 1000)

                    terminal_width = 10
                    logger.info("-" * terminal_width)
                    logger.info("🤖Grounding Output🤖")

                    for line in output.split('\n'):
                        logger.info("%s", line)
                    # logger.info(output)
                    pred_element, pred_action, pred_value = postprocess_action_lmm(output)
                    if len(pred_element) in [1, 2]:
                        element_id = get_index_from_option_name(pred_element)
                    else:
                        element_id = -1

                    # Process the elements
                    if (0 <= element_id < len(candidate_ids) and pred_action.strip() in ["CLICK", "SELECT", "TYPE",
                                                                                         "PRESS ENTER", "HOVER",
                                                                                         "TERMINATE"]):
                        # Map chosen option (0..len(candidate_ids)-1) back to original element index
                        target_element = elements[candidate_ids[element_id]]
                        # choices is a list of textual strings; use the selected one directly
                        target_element_text = choices[element_id]
                        target_action = pred_action
                        target_value = pred_value
                        new_action += "[" + target_element[2] + "]" + " "
                        new_action += target_element[1] + " -> " + target_action
                        if target_action.strip() in ["SELECT", "TYPE"]:
                            new_action += ": " + target_value
                        got_one_answer = True
                        break
                    elif pred_action.strip() in ["PRESS ENTER", "TERMINATE"]:
                        target_element = pred_action
                        target_element_text = target_element
                        target_action = pred_action
                        target_value = pred_value
                        new_action += target_action
                        if target_action.strip() in ["SELECT", "TYPE"]:
                            new_action += ": " + target_value
                        got_one_answer = True
                        break
                    else:
                        # Fuzzy fallback: use the same one-turn output text to infer likely target
                        def _extract_plan_targets(text: str):
                            import re as _re
                            cands = []
                            cands += _re.findall(r'"([^"]{2,80})"', text)
                            cands += _re.findall(r"'([^']{2,80})'", text)
                            # Common verbs
                            verb = _re.search(r"\b(click|select|type|press|goto|go to)\b\s+([^\n\.;:,]{2,80})", text, _re.I)
                            if verb:
                                cands.append(verb.group(2))
                            # Clean
                            return [s.strip() for s in cands if s.strip()]

                        def _parse_action_from_plan(text: str):
                            tl = text.lower()
                            if "press enter" in tl:
                                return "PRESS ENTER", ""
                            if "type" in tl:
                                # extract quoted value
                                import re as _re
                                m = _re.search(r"type\s+\"([^\"]+)\"", text, _re.I)
                                return "TYPE", (m.group(1) if m else "")
                            if "select" in tl:
                                return "SELECT", ""
                            if "click" in tl:
                                return "CLICK", ""
                            return "CLICK", ""

                        def _clean_choice(s: str) -> str:
                            # Remove coords and DOM trail for similarity
                            base = s.split(" | DOM:", 1)[0]
                            # Drop leading coordinate + space
                            if ") " in base:
                                base = base.split(") ", 1)[1]
                            # Remove tags like <a ...> and </a>
                            import re as _re
                            base = _re.sub(r"<[^>]+>", " ", base)
                            return base.strip().lower()

                        plan_targets = _extract_plan_targets(output)
                        if plan_targets and choices:
                            best = (-1.0, None)
                            for idx, txt in enumerate(choices):
                                c = _clean_choice(txt)
                                for t in plan_targets:
                                    score = SequenceMatcher(None, c, t.lower()).ratio()
                                    if score > best[0]:
                                        best = (score, idx)
                            if best[1] is not None and best[0] >= 0.45:
                                fb_idx = best[1]
                                target_element = elements[candidate_ids[fb_idx]]
                                target_element_text = choices[fb_idx]
                                action_guess, value_guess = _parse_action_from_plan(output)
                                target_action = action_guess
                                target_value = value_guess
                                new_action += "[" + target_element[2] + "]" + " "
                                new_action += target_element[1] + " -> " + target_action
                                if target_action in ["SELECT", "TYPE"] and target_value:
                                    new_action += ": " + target_value
                                got_one_answer = True
                                break
                    # Log per-batch metrics
                    try:
                        prompt_text = "\n".join(str(p) for p in (prompt or []))
                        approx_tokens = int(len(prompt_text) / 4)
                        logger.info(f"metrics: scan_ms={t_scan_ms} ss_ms={ss_ms} img_bytes={img_bytes} llm_ms={llm_ms} approx_tokens={approx_tokens}")
                    except Exception:
                        pass

                if got_one_answer:
                    terminal_width = 10
                    logger.info("-" * terminal_width)
                    logger.info("🤖Browser Operation🤖")
                    logger.info(f"Target Element: {target_element_text}", )
                    logger.info(f"Target Action: {target_action}", )
                    logger.info(f"Target Value: {target_value}", )

                    if monitor:
                        logger.info(
                            f"----------\nShould I execute the above action? [Y/n/i/e].\nY/n: Accept or reject this action.\ni(Intervene): Reject this action, and pause for human intervention.\ne(Exit): Terminate the program and save results.")
                        monitor_input = await ainput()
                        logger.info("Monitor Command: " + monitor_input)
                        if monitor_input in ["n", "N", "No", "no"]:
                            monitor_signal = "reject"
                            target_element = []
                        elif monitor_input in ["e", "exit", "Exit"]:
                            monitor_signal = "exit"
                        elif monitor_input in ["i", "intervene", 'intervention']:
                            monitor_signal = "pause"
                            target_element = []
                        else:
                            valid_op_count += 1
                else:
                    no_op_count += 1
                    target_element = []

                try:
                    if monitor_signal == 'exit':
                        raise Exception("human supervisor manually made it exit.")
                    if no_op_count >= max_continuous_no_op:
                        raise Exception(f"no executable operations for {max_continuous_no_op} times.")
                    elif time_step >= max_op:
                        raise Exception(f"the agent reached the step limit {max_op}.")
                    elif target_action == "TERMINATE":
                        raise Exception("The model determined a completion.")

                    # Perform browser action with PlayWright
                    # The code is complex to handle all kinds of cases in execution
                    # It's ugly, but it works, so far
                    selector = None
                    fail_to_execute = False
                    try:
                        if target_element == []:
                            pass
                        else:
                            if not target_element in ["PRESS ENTER", "TERMINATE"]:
                                selector = target_element[-2]
                                if dev_mode:
                                    logger.info(target_element)
                                try:
                                    await selector.scroll_into_view_if_needed(timeout=3000)
                                    if highlight:
                                        await selector.highlight()
                                        await asyncio.sleep(2.5)
                                except Exception as e:
                                    pass

                        if selector:
                            valid_op_count += 1
                            if target_action == "CLICK":
                                js_click = True
                                try:
                                    if target_element[-1] in ["select", "input"]:
                                        logger.info("Try performing a CLICK")
                                        await selector.evaluate("element => element.click()", timeout=10000)
                                        js_click = False
                                    else:
                                        await selector.click(timeout=10000)
                                except Exception as e:
                                    try:
                                        if not js_click:
                                            logger.info("Try performing a CLICK")
                                            await selector.evaluate("element => element.click()", timeout=10000)
                                        else:
                                            raise Exception(e)
                                    except Exception as ee:
                                        try:
                                            logger.info("Try performing a HOVER")
                                            await selector.hover(timeout=10000)
                                            new_action = new_action.replace("CLICK",
                                                                            f"Failed to CLICK because {e}, did a HOVER instead")
                                        except Exception as eee:
                                            new_action = new_action.replace("CLICK", f"Failed to CLICK because {e}")
                                            no_op_count += 1
                            elif target_action == "TYPE":
                                try:
                                    try:
                                        logger.info("Try performing a \"press_sequentially\"")
                                        await selector.clear(timeout=10000)
                                        await selector.fill("", timeout=10000)
                                        await selector.press_sequentially(target_value, timeout=10000)
                                    except Exception as e0:
                                        await selector.fill(target_value, timeout=10000)
                                except Exception as e:
                                    try:
                                        if target_element[-1] in ["select"]:
                                            logger.info("Try performing a SELECT")
                                            selected_value = await select_option(selector, target_value)
                                            new_action = new_action.replace("TYPE",
                                                                            f"Failed to TYPE \"{target_value}\" because {e}, did a SELECT {selected_value} instead")
                                        else:
                                            raise Exception(e)
                                    except Exception as ee:
                                        js_click = True
                                        try:
                                            if target_element[-1] in ["select", "input"]:
                                                logger.info("Try performing a CLICK")
                                                await selector.evaluate("element => element.click()", timeout=10000)
                                                js_click = False
                                            else:
                                                logger.info("Try performing a CLICK")
                                                await selector.click(timeout=10000)
                                            new_action = "[" + target_element[2] + "]" + " "
                                            new_action += target_element[
                                                              1] + " -> " + f"Failed to TYPE \"{target_value}\" because {e}, did a CLICK instead"
                                        except Exception as eee:
                                            try:
                                                if not js_click:
                                                    if dev_mode:
                                                        logger.info(eee)
                                                    logger.info("Try performing a CLICK")
                                                    await selector.evaluate("element => element.click()", timeout=10000)
                                                    new_action = "[" + target_element[2] + "]" + " "
                                                    new_action += target_element[
                                                                      1] + " -> " + f"Failed to TYPE \"{target_value}\" because {e}, did a CLICK instead"
                                                else:
                                                    raise Exception(eee)
                                            except Exception as eeee:
                                                try:
                                                    logger.info("Try performing a HOVER")
                                                    await selector.hover(timeout=10000)
                                                    new_action = "[" + target_element[2] + "]" + " "
                                                    new_action += target_element[
                                                                      1] + " -> " + f"Failed to TYPE \"{target_value}\" because {e}, did a HOVER instead"
                                                except Exception as eee:
                                                    new_action = "[" + target_element[2] + "]" + " "
                                                    new_action += target_element[
                                                                      1] + " -> " + f"Failed to TYPE \"{target_value}\" because {e}"
                                                    no_op_count += 1
                            elif target_action == "SELECT":
                                try:
                                    logger.info("Try performing a SELECT")
                                    selected_value = await select_option(selector, target_value)
                                    new_action = new_action.replace(f"{target_value}", f"{selected_value}")
                                except Exception as e:
                                    try:
                                        if target_element[-1] in ["input"]:
                                            try:
                                                logger.info("Try performing a \"press_sequentially\"")
                                                await selector.clear(timeout=10000)
                                                await selector.fill("", timeout=10000)
                                                await selector.press_sequentially(target_value, timeout=10000)
                                            except Exception as e0:
                                                await selector.fill(target_value, timeout=10000)
                                            new_action = new_action.replace("SELECT",
                                                                            f"Failed to SELECT \"{target_value}\" because {e}, did a TYPE instead")
                                        else:
                                            raise Exception(e)
                                    except Exception as ee:
                                        js_click = True
                                        try:
                                            if target_element[-1] in ["select", "input"]:
                                                logger.info("Try performing a CLICK")
                                                await selector.evaluate("element => element.click()", timeout=10000)
                                                js_click = False
                                            else:
                                                await selector.click(timeout=10000)
                                            new_action = "[" + target_element[2] + "]" + " "
                                            new_action += target_element[
                                                              1] + " -> " + f"Failed to SELECT \"{target_value}\" because {e}, did a CLICK instead"
                                        except Exception as eee:

                                            try:
                                                if not js_click:
                                                    logger.info("Try performing a CLICK")
                                                    await selector.evaluate("element => element.click()", timeout=10000)
                                                    new_action = "[" + target_element[2] + "]" + " "
                                                    new_action += target_element[
                                                                      1] + " -> " + f"Failed to SELECT \"{target_value}\" because {e}, did a CLICK instead"
                                                else:
                                                    raise Exception(eee)
                                            except Exception as eeee:
                                                try:
                                                    logger.info("Try performing a HOVER")
                                                    await selector.hover(timeout=10000)
                                                    new_action = "[" + target_element[2] + "]" + " "
                                                    new_action += target_element[
                                                                      1] + " -> " + f"Failed to SELECT \"{target_value}\" because {e}, did a HOVER instead"
                                                except Exception as eee:
                                                    new_action = "[" + target_element[2] + "]" + " "
                                                    new_action += target_element[
                                                                      1] + " -> " + f"Failed to SELECT \"{target_value}\" because {e}"
                                                    no_op_count += 1
                            elif target_action == "HOVER":
                                try:
                                    logger.info("Try performing a HOVER")
                                    await selector.hover(timeout=10000)
                                except Exception as e:
                                    try:
                                        await selector.click(timeout=10000)
                                        new_action = new_action.replace("HOVER",
                                                                        f"Failed to HOVER because {e}, did a CLICK instead")
                                    except:
                                        js_click = True
                                        try:
                                            if target_element[-1] in ["select", "input"]:
                                                logger.info("Try performing a CLICK")
                                                await selector.evaluate("element => element.click()", timeout=10000)
                                                js_click = False
                                            else:
                                                await selector.click(timeout=10000)
                                            new_action = "[" + target_element[2] + "]" + " "
                                            new_action += target_element[
                                                              1] + " -> " + f"Failed to HOVER because {e}, did a CLICK instead"
                                        except Exception as eee:
                                            try:
                                                if not js_click:
                                                    logger.info("Try performing a CLICK")
                                                    await selector.evaluate("element => element.click()", timeout=10000)
                                                    new_action = "[" + target_element[2] + "]" + " "
                                                    new_action += target_element[
                                                                      1] + " -> " + f"Failed to HOVER because {e}, did a CLICK instead"
                                                else:
                                                    raise Exception(eee)
                                            except Exception as eeee:
                                                new_action = "[" + target_element[2] + "]" + " "
                                                new_action += target_element[
                                                                  1] + " -> " + f"Failed to HOVER because {e}"
                                                no_op_count += 1
                            elif target_action == "PRESS ENTER":
                                try:
                                    logger.info("Try performing a PRESS ENTER")
                                    await selector.press('Enter')
                                except Exception as e:
                                    await selector.click(timeout=10000)
                                    await session_control.active_page.keyboard.press('Enter')
                        elif monitor_signal == "pause":
                            logger.info(
                                "Pause for human intervention. Press Enter to continue. You can also enter your message here, which will be included in the action history as a human message.")
                            human_intervention = await ainput()
                            if human_intervention:
                                human_intervention = f" Human message: {human_intervention}"
                            raise Exception(
                                f"the human supervisor rejected this operation and may have taken some actions.{human_intervention}")
                        elif monitor_signal == "reject":
                            raise Exception("the human supervisor rejected this operation.")
                        elif target_element == "PRESS ENTER":
                            logger.info("Try performing a PRESS ENTER")
                            await session_control.active_page.keyboard.press('Enter')
                        no_op_count = 0
                        try:
                            await session_control.active_page.wait_for_load_state('load')
                        except Exception as e:
                            pass
                    except Exception as e:
                        if target_action not in ["TYPE", "SELECT"]:
                            new_action = f"Failed to {target_action} {target_element_text} because {e}"

                        else:
                            new_action = f"Failed to {target_action} {target_value} for {target_element_text} because {e}"
                        fail_to_execute = True

                    if new_action == "" or fail_to_execute:
                        if new_action == "":
                            new_action = "No Operation"
                        if monitor_signal not in ["pause", "reject"]:
                            no_op_count += 1
                    taken_actions.append(new_action)
                    if not session_control.context.pages:
                        await session_control.context.new_page()
                        try:
                            await session_control.active_page.goto(confirmed_website_url, wait_until="load")
                        except Exception as e:
                            pass

                    if monitor_signal == 'pause':
                        pass
                    else:
                        await asyncio.sleep(0.3)
                    if dev_mode:
                        logger.info(f"current active page: {session_control.active_page}")

                        # await session_control.context.new_page()
                        # try:
                        #     await session_control.active_page.goto("https://www.bilibili.com/", wait_until="load")
                        # except Exception as e:
                        #     pass
                        logger.info("All pages")
                        logger.info(session_control.context.pages)
                        logger.info("-" * 10)
                    try:
                        await session_control.active_page.wait_for_load_state('load')
                    except Exception as e:
                        if dev_mode:
                            logger.info(e)
                    if tracing:
                        logger.info("Save playwright trace file")
                        await session_control.context.tracing.stop_chunk(
                            path=f"{os.path.join(main_result_path, 'playwright_traces', f'{time_step}.zip')}")
                except Exception as e:
                    logger.info("=" * 10)
                    logger.info(f"Decide to terminate because {e}")
                    logger.info("Action History:")

                    for action in taken_actions:
                        logger.info(action)
                    logger.info("")

                    if tracing:
                        logger.info("Save playwright trace file")
                        await session_control.context.tracing.stop_chunk(
                            path=f"{os.path.join(main_result_path, 'playwright_traces', f'{time_step}.zip')}")

                    success_or_not = ""
                    if valid_op_count == 0:
                        success_or_not = "0"
                    logger.info(f"Write results to json file: {os.path.join(main_result_path, 'result.json')}")
                    final_json = {"confirmed_task": confirmed_task, "website": confirmed_website,
                                  "task_id": task_id, "success_or_not": success_or_not,
                                  "num_step": len(taken_actions), "action_history": taken_actions, "exit_by": str(e)}

                    with open(os.path.join(main_result_path, 'result.json'), 'w', encoding='utf-8') as file:
                        json.dump(final_json, file, indent=4)

                    if monitor:
                        logger.info("Wait for human inspection. Directly press Enter to exit")
                        monitor_input = await ainput()

                    logger.info("Close browser context")
                    logger.removeHandler(log_fh)
                    logger.removeHandler(console_handler)
                    close_context = session_control.context
                    session_control.context = None
                    await close_context.close()

                    complete_flag = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config_path", help="Path to the TOML configuration file.", type=str, metavar='config',
                        default=f"{os.path.join('config', 'demo_mode.toml')}")
    args = parser.parse_args()

    # Load configuration file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = None
    resolved = None
    try:
        # Priority: absolute path -> CWD-relative -> module-relative
        if os.path.isabs(args.config_path):
            candidate = args.config_path
            if os.path.exists(candidate):
                resolved = candidate
        else:
            # Try from current working directory first
            cwd_candidate = os.path.abspath(args.config_path)
            if os.path.exists(cwd_candidate):
                resolved = cwd_candidate
            else:
                # Fallback 1: module base_dir join
                base_candidate = os.path.join(base_dir, args.config_path)
                if os.path.exists(base_candidate):
                    resolved = base_candidate
                else:
                    # Fallback 2: treat as package-relative path under repo src
                    # e.g., 'seeact/config/demo_mode.toml' when running from repo root
                    repo_src = Path(base_dir).parent  # .../src
                    pkg_candidate = repo_src / args.config_path
                    if pkg_candidate.exists():
                        resolved = str(pkg_candidate)

        if not resolved:
            raise FileNotFoundError

        # Use TOML compat loader
        config = _load_toml(resolved)
        print(f"Configuration File Loaded - {resolved}")
    except FileNotFoundError:
        print(f"Error: File '{args.config_path}' not found.")
    except _TomlDecodeError:
        print(f"Error: File '{args.config_path}' is not a valid TOML file.")

    if config is None:
        raise SystemExit(1)

    asyncio.run(main(config, base_dir))
