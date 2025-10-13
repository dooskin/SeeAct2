# TODO: add more models to enum and make the other engine factories output the special response object
from dataclasses import dataclass
from enum import Enum
import os
import time

import backoff
import openai
from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
)
import requests
from dotenv import load_dotenv
import litellm
import base64
import logging

logger = logging.getLogger(__name__)  # runner module logger
logger.setLevel(logging.DEBUG)
#  Add a file handler for the runner logger
f_handler = logging.FileHandler('inference_engine.log')
f_handler.setLevel(logging.DEBUG)
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)

# Add a console handler for the runner logger
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
# Include logger name in the console output
c_format = logging.Formatter('inference_engine.py: %(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)


def load_openai_api_key():
    load_dotenv()
    assert (
            os.getenv("OPENAI_API_KEY") is not None
    ), "must set OPENAI_API_KEY in the environment"
    return os.getenv("OPENAI_API_KEY")


def load_gemini_api_key():
    load_dotenv()
    assert (
            os.getenv("GEMINI_API_KEY") is not None
    ), "must set GEMINI_API_KEY in the environment"
    return os.getenv("GEMINI_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def engine_factory(api_key=None, model=None, **kwargs):
    """Factory to build an inference engine based on model/provider.

    Enhancements:
    - If an OpenAI-compatible base URL is provided (via [openai].base_url or OPENAI_BASE_URL),
      accept any `model` string and use the official OpenAI client pointed at that base URL.
    - Backwards compatible with explicit OpenAI model names when no base_url is set.
    """
    model = (model or "").lower()
    # Detect OpenAI-compatible endpoint via config or environment
    base_url = kwargs.pop("base_url", None) or os.getenv("OPENAI_BASE_URL")

    if base_url or model in ["gpt-4-vision-preview", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]:
        # Ensure API key is available
        load_openai_api_key()
        # Pass base_url to engine (client reads env if None)
        return OpenAIEngine(model=model or "openai-compatible", base_url=base_url, **kwargs)
    elif model in ["gemini-1.5-pro-latest", "gemini-1.5-flash"]:
        load_gemini_api_key()
        model=f"gemini/{model}"
        return GeminiEngine(model=model, **kwargs)
    elif model == "llava":
        model="llava"
        return OllamaEngine(model=model, **kwargs)
    raise Exception(f"Unsupported model: {model}, currently supported models: \
                    gpt-4-vision-preview, gpt-4-turbo, gpt-4o, , gpt-4o-mini, gemini-1.5-pro-latest, llava")

class OOBLanguageModel(Enum):
    # This will be used to compute cost of each API call
    # Open AI
    GPT4Vision = "gpt-4-vision-preview"
    GPT4Turbo = "gpt-4-turbo"
    GPT4o = "gpt-4o"
    GPT4oMini = "gpt-4o-mini"
    
    # Gemini
    Gemini1_5Pro = "gemini-1.5-pro-latest"
    Gemini1_5Flash = "gemini-1.5-flash"
    
    # llama
    UNKNOWN = "unknown"

@dataclass
class SquooshEngineResponse:
    message: str
    tokens_prompt: int = -1
    tokens_completion: int = -1
    model: OOBLanguageModel = OOBLanguageModel.UNKNOWN
    includes_image: bool = False

class Engine:
    def __init__(
            self,
            stop=["\n\n"],
            rate_limit=-1,
            model=None,
            temperature=0,
            **kwargs,
    ) -> None:
        """
            Base class to init an engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        self.time_slots = [0]
        self.stop = stop
        self.temperature = temperature
        self.model = model
        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.time_slots)
        self.current_key_idx = 0
        print(f"Initializing model {self.model}")        

    def tokenize(self, input):
        return self.tokenizer(input)


class OllamaEngine(Engine):
    def __init__(self, **kwargs) -> None:
        """
            Init an Ollama engine
            To use Ollama, dowload and install Ollama from https://ollama.com/
            After Ollama start, pull llava with command: ollama pull llava
        """
        super().__init__(**kwargs)
        self.api_url = "http://localhost:11434/api/chat"


    def generate(self, prompt: list = None, max_new_tokens=4096, temperature=None, model=None, image_path=None,
                 ouput_0=None, turn_number=0, image_detail: str = "auto", **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.time_slots)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            wait_time = self.next_avil_time[self.current_key_idx] - start_time
            print(f"Wait {wait_time} for rate limitting")
            time.sleep(wait_time)
        prompt0, prompt1, prompt2 = prompt

        base64_image = encode_image(image_path)
        if turn_number == 0:
            # Assume one turn dialogue
            prompt_input = [
                {"role": "assistant", "content": prompt0},
                {"role": "user", "content": prompt1, "images": [f"{base64_image}"]},
            ]
        elif turn_number == 1:
            prompt_input = [
                {"role": "assistant", "content": prompt0},
                {"role": "user", "content": prompt1, "images": [f"{base64_image}"]},
                {"role": "assistant", "content": f"\n\n{ouput_0}"},
                {"role": "user", "content": prompt2}, 
            ]

        options = {"temperature": self.temperature, "num_predict": max_new_tokens}
        data = {
            "model": self.model,
            "messages": prompt_input,
            "options": options,
            "stream": False,
        }
        _request = {
            "url": f"{self.api_url}",
            "json": data,
        }
        response = requests.post(**_request)  # type: ignore
        if response.status_code != 200:
            raise Exception(f"Ollama API Error: {response.status_code}, {response.text}")
        response_json = response.json()
        return response_json["message"]["content"]


class GeminiEngine(Engine):
    def __init__(self, **kwargs) -> None:
        """
            Init a Gemini engine
            To use this engine, please provide the GEMINI_API_KEY in the environment
            Supported Model             Rate Limit
            gemini-1.5-pro-latest    	2 queries per minute, 1000 queries per day
        """
        super().__init__(**kwargs)


    def generate(self, prompt: list = None, max_new_tokens=4096, temperature=None, model=None, image_path=None,
                 ouput_0=None, turn_number=0, image_detail: str = "auto", **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.time_slots)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            wait_time = self.next_avil_time[self.current_key_idx] - start_time
            print(f"Wait {wait_time} for rate limitting")
        prompt0, prompt1, prompt2 = prompt
        litellm.set_verbose=True

        base64_image = encode_image(image_path)
        if turn_number == 0:
            # Assume one turn dialogue
            prompt_input = [
                {"role": "system", "content": prompt0},
                {"role": "user",
                 "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url": image_path,
                                                                                                    "detail": "high"},
                                                                }]},
            ]
        elif turn_number == 1:
            prompt_input = [
                {"role": "system", "content": prompt0},
                {"role": "user",
                 "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url": image_path,
                                                                                                    "detail": "high"}, 
                                                                }]},
                {"role": "assistant", "content": [{"type": "text", "text": f"\n\n{ouput_0}"}]},
                {"role": "user", "content": [{"type": "text", "text": prompt2}]}, 
            ]
        response = litellm.completion(
            model=model if model else self.model,
            messages=prompt_input,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature if temperature else self.temperature,
            **kwargs,
        )
        return [choice["message"]["content"] for choice in response.choices][0]


class OpenAIEngine(Engine):
    def __init__(self, base_url: str | None = None, **kwargs) -> None:
        """
            Init an OpenAI GPT/Codex engine
            To find your OpenAI API key, visit https://platform.openai.com/api-keys
        """
        super().__init__(**kwargs)
        try:
            # Prefer the official OpenAI client for OpenAI models
            from openai import OpenAI as _OpenAIClient  # type: ignore
            # Honor explicit base_url if provided; otherwise the client will read OPENAI_BASE_URL
            self._client = _OpenAIClient(base_url=base_url) if base_url else _OpenAIClient()
        except Exception:
            self._client = None

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError), logger=logger
    )
    def generate(self, prompt: list = None, max_new_tokens=4096, temperature=None, model=None, image_path=None,
                 ouput_0=None, turn_number=0, image_detail: str = "auto", **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.time_slots)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)
        prompt0, prompt1, prompt2 = prompt
        base64_image = encode_image(image_path) if image_path else None

        # Prepare OpenAI chat messages with multimodal content
        def _mm_text(txt: str):
            return {"type": "text", "text": txt}

        def _mm_image_b64(b64: str):
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": image_detail}}

        if turn_number == 0:
            user_content = [_mm_text(prompt1)] + ([_mm_image_b64(base64_image)] if base64_image else [])
            messages = [
                {"role": "system", "content": [_mm_text(prompt0)]},
                {"role": "user", "content": user_content},
            ]
        else:
            user_content = [_mm_text(prompt1)] + ([_mm_image_b64(base64_image)] if base64_image else [])
            messages = [
                {"role": "system", "content": [_mm_text(prompt0)]},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [_mm_text(f"\n\n{ouput_0}")]},
                {"role": "user", "content": [_mm_text(prompt2)]},
            ]

        # Prefer official client; fallback to litellm if client unavailable
        if self._client is not None:
            resp = self._client.chat.completions.create(
                model=model if model else self.model,
                messages=messages,  # type: ignore
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature if temperature else self.temperature,
            )
            response_ = SquooshEngineResponse(
                message=resp.choices[0].message.content,
                tokens_prompt=resp.usage.prompt_tokens,
                tokens_completion=resp.usage.completion_tokens,
                model=OOBLanguageModel(self.model) if self.model in OOBLanguageModel._value2member_map_ else None,
                includes_image=bool(base64_image)
            )
            return response_
        else:
            response = litellm.completion(
                model=model if model else self.model,
                messages=messages,  # type: ignore
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature if temperature else self.temperature,
                **kwargs,
            )
            
            response_ = SquooshEngineResponse(
                message=response.choices[0].message.content,
                tokens_prompt=response.usage.prompt_tokens,
                tokens_completion=response.usage.completion_tokens,
                model=OOBLanguageModel(self.model) if self.model in OOBLanguageModel._value2member_map_ else None,
                includes_image=bool(base64_image)
            )
            
            return response_


class OpenaiEngine_MindAct(Engine):
    def __init__(self, **kwargs) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        super().__init__(**kwargs)
    #
    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
    )
    def generate(self, prompt, max_new_tokens=50, temperature=0, model=None, **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.time_slots)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)
        if isinstance(prompt, str):
            # Assume one turn dialogue
            prompt = [
                {"role": "user", "content": prompt},
            ]
        response = litellm.completion(
            model=model if model else self.model,
            messages=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        if self.request_interval > 0:
            self.next_avil_time[self.current_key_idx] = (
                    max(start_time, self.next_avil_time[self.current_key_idx])
                    + self.request_interval
            )
        return [choice["message"]["content"] for choice in response["choices"]]

# Backward-compatible alias (legacy code imports OpenaiEngine)
OpenaiEngine = OpenAIEngine
