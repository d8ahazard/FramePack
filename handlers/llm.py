import json
import os
import base64
import traceback
import requests

from openai import OpenAI
import anthropic
from groq import Groq
from handlers.path import app_path
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
import logging

logger = logging.getLogger(__name__)

# Load API keys
key_file = os.path.join(app_path, "apikeys.json")
supported_providers = ["openai", "anthropic", "deepseek", "gemini", "groq", "openwebui"]
keys_validated = False
validated_keys = {}
invalid_keys = {}
active_keys = {}
observer = None

IMAGE_PROMPT = """
You are an assistant that writes short, motion-focused prompts for animating images.

When the user sends an image, respond with a single, concise prompt describing visual motion (such as human activity, moving objects, or camera movements). Focus only on how the scene could come alive and become dynamic using brief phrases.

Larger and more dynamic motions (like dancing, jumping, running, etc.) are preferred over smaller or more subtle ones (like standing still, sitting, etc.).

Describe subject, then motion, then other things. For example: "The girl dances gracefully, with clear movements, full of charm."

If there is something that can dance (like a man, girl, robot, etc.), then prefer to describe it as dancing.

Stay in a loop: one image in, one motion prompt out. Do not explain, ask questions, or generate multiple options."""

def validate_llm_key(provider: str, key: str, refresh: bool = False) -> bool:
    if provider in validated_keys and not refresh:
        return True
    if provider in invalid_keys and not refresh:
        return False
    if key == "" or key is None:
        return False
    try:
        if provider == "openai":
            client = OpenAI(api_key=key)
            _ = client.models.list()
            print("OpenAI key validated")
            return True
        if provider == "anthropic":
            client = anthropic.Anthropic(api_key=key)
            client.models.list()
            return True
        if provider == "deepseek":
            r = requests.get("https://api-docs.deepseek.com/v1/models", headers={"Authorization": f"Bearer {key}"})
            return r.status_code == 200
        if provider == "gemini":
            r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}")
            return r.status_code == 200
        if provider == "groq":
            Groq(api_key=key).chat.completions.list()
            return True
        if provider == "openwebui":
            base = os.getenv("OPENWEBUI_API_URL", "http://localhost:3000")
            r = requests.get(f"{base}/api/models", headers={"Authorization": f"Bearer {key}"})
            return r.status_code == 200
    except Exception as e:
        print(f"Error validating key for {provider}: {e}")
        pass
    return False

class MyEventHandler(FileSystemEventHandler):
    def on_any_event(self, event: FileSystemEvent) -> None:
        global active_keys
        print(event)
        # Only re-validate keys if the file has been modified
        if event.event_type == "modified":
            try:
                with open(key_file, "r") as f:
                    keys = json.load(f)
            except FileNotFoundError:
                keys = {}

            active_keys = {}
            for provider, key in keys.items():
                if provider not in supported_providers:
                    raise ValueError(f"Unsupported provider: {provider}")
                if key == "" or key is None:
                    env_var = f"{provider.upper()}_API_KEY"
                    key = os.getenv(env_var, None)
                if key:
                    active_keys[provider] = key

            for provider in active_keys:
                validate_llm_key(provider, active_keys[provider], refresh=True) 




async def startup_event():
    global active_keys, keys_validated, observer
    try:
        with open(key_file, "r") as f:
            keys = json.load(f)
    except FileNotFoundError:
        keys = {}

    active_keys = {}
    for provider, key in keys.items():
        if provider not in supported_providers:
            raise ValueError(f"Unsupported provider: {provider}")
        if key == "" or key is None:
            env_var = f"{provider.upper()}_API_KEY"
            key = os.getenv(env_var, None)
        if key:
            active_keys[provider] = key

    # Key validation
    if not keys_validated:
        for provider, key in active_keys.items():
            if validate_llm_key(provider, key):
                validated_keys[provider] = key
            else:
                invalid_keys[provider] = key
        keys_validated = True

    observer = Observer()
    observer.schedule(MyEventHandler(), key_file, recursive=False)
    observer.start()

async def shutdown_event():
    global observer
    if observer:
        observer.stop()
        observer.join()
    logger.info("Shutdown event complete for LLM")

# Caption functions
def caption_openai(image_path: str, prompt: str) -> str:
    client = OpenAI(api_key=active_keys["openai"])
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url":f"data:image/jpeg;base64,{img_b64}", 
                    "detail": "high"
                    }}
            ]
        }],
    )
    return response.choices[0].message.content

def caption_anthropic(image_path: str, prompt: str) -> str:
    client = anthropic.Client(active_keys["anthropic"])
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]
    resp = client.messages.create(
        model="claude-3.7-sonnet",
        messages=messages,
        max_tokens_to_sample=1000
    )
    return resp.completion

def caption_deepseek(image_path: str, prompt: str) -> str:
    key = active_keys["deepseek"]
    url = "https://api-docs.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "model": "deepseek-r1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "data": img_b64}}
                ]
            }
        ]
    }
    r = requests.post(url, json=payload, headers=headers)
    return r.json()["choices"][0]["message"]["content"]

def caption_gemini(image_path: str, prompt: str) -> str:
    key = active_keys["gemini"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateMessage?key={key}"
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "prompt": {
            "parts": [
                {"text": prompt},
                {"content": img_b64, "mime_type": "image/jpeg"}
            ]
        },
        "model": "gemini-2.0-flash",
    }
    resp = requests.post(url, json=payload)
    return resp.json()["candidates"][0]["message"]["content"]

def caption_groq(image_path: str, prompt: str) -> str:
    client = Groq(api_key=active_keys["groq"])
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64
                    }
                }
            ]
        }
    ]
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages
    )
    return resp.choices[0].message.content

def caption_openwebui(image_path: str, prompt: str) -> str:
    key = active_keys["openwebui"]
    base = os.getenv("OPENWEBUI_API_URL", "http://localhost:3000")
    url = f"{base}/api/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "data": img_b64}}
                ]
            }
        ]
    }
    r = requests.post(url, json=payload, headers=headers)
    return r.json()["choices"][0]["message"]["content"]

def caption_auto(image_path: str, prompt: str = IMAGE_PROMPT) -> str:
    for prov in supported_providers:
        if prov in active_keys:
            try:
                cap = globals()[f"caption_{prov}"](image_path, prompt)
                return {"provider": prov, "caption": cap}
            except:
                traceback.print_exc()
                continue
    logger.error(f"No available LLM providers to caption the image: {image_path}")
    return None


# Register endpoints
def register_api_endpoints(app):
    api_tag = __name__.split(".")[-1].title().replace("_", " ")
    @app.get("/api/llm/providers", tags=[api_tag])
    def list_llm_models():
        return [
            {"provider": prov, "valid": validate_llm_key(prov, active_keys.get(prov, ""))}
            for prov in supported_providers
        ]

    @app.post("/api/llm/caption", tags=[api_tag])
    def caption_image(provider: str, image_path: str, prompt: str):
        if provider not in active_keys:
            raise ValueError(f"No valid key for provider '{provider}'")
        func = globals().get(f"caption_{provider}")
        return {"provider": provider, "caption": func(image_path, prompt)}

    @app.post("/api/llm/auto_caption", tags=[api_tag])
    def auto_caption(image_path: str, prompt: str = IMAGE_PROMPT):
        for prov in supported_providers:
            if prov in active_keys:
                try:
                    print(f"Captioning with {prov}")
                    cap = globals()[f"caption_{prov}"](image_path, prompt)
                    return {"provider": prov, "caption": cap}
                except:
                    traceback.print_exc()
                    continue
        raise ValueError("No available LLM providers to caption the image")
