
"""
Head tilt -> Handy transcribes the question -> AI answers.

Answer modes:
  - api      — query a provider (OpenRouter / OpenAI / Anthropic / DeepSeek)
  - desktop  — paste the question into the Claude desktop app (macOS only)

Commands:
  python3 app.py          run (first launch starts the setup wizard)
  python3 app.py setup    reconfigure (mode, provider, key)
  python3 app.py models   change fast/accurate models

Requirements:
  pip install mediapipe opencv-python
"""

import collections
import concurrent.futures
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# --- detection ---
CAM_INDEX = 0
TILT_ON_DEG = 15.0   # tilt angle that arms the gesture (degrees)
TILT_OFF_DEG = 7.0   # return-to-neutral threshold that fires it
BASELINE_ALPHA = 0.05
COOLDOWN_SEC = 1.2
SHOW_WINDOW = True
LEFT_EYE = 33
RIGHT_EYE = 263
MODEL_NAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
CLIPBOARD_TIMEOUT_SEC = 45
NEW_CHAT_PER_QUERY = False  # desktop mode: open a new chat per question

CONFIG_PATH = pathlib.Path(__file__).resolve().with_name("config.json")

FAST_PROMPT = (
    "Отвечай на вопрос сразу и по существу: сначала главное одной-двумя фразами, "
    "затем минимум необходимых деталей. Коротко, конкретно, без воды и вступлений. "
    "Только обычный текст, без Markdown, списков и заголовков. Отвечай на языке вопроса."
)
ACCURATE_PROMPT = (
    "Дай точный и достоверный ответ на вопрос: сначала суть, затем краткое обоснование "
    "или ключевые шаги, если они нужны для понимания. Не выдумывай, не раздувай ответ — "
    "лучше плотно и по делу, чем длинно. "
    "Только обычный текст, без Markdown, списков и заголовков. Отвечай на языке вопроса."
)

# --- providers ---
# wire: "openai" = chat/completions with Bearer key,
#       "anthropic" = /v1/messages with x-api-key.
PROVIDERS = {
    "openrouter": {
        "title": "OpenRouter (любые модели через один ключ)",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "wire": "openai",
        "key_env": "OPENROUTER_API_KEY",
        "key_hint": "https://openrouter.ai/settings/keys",
        "fast": "google/gemini-2.5-flash",
        "accurate": "anthropic/claude-opus-4.8",
        "suggest": [
            "google/gemini-2.5-flash",
            "anthropic/claude-opus-4.8",
            "anthropic/claude-sonnet-4.6",
            "openai/gpt-5-mini",
            "deepseek/deepseek-chat",
        ],
    },
    "openai": {
        "title": "OpenAI",
        "url": "https://api.openai.com/v1/chat/completions",
        "wire": "openai",
        "key_env": "OPENAI_API_KEY",
        "key_hint": "https://platform.openai.com/api-keys",
        "fast": "gpt-5-mini",
        "accurate": "gpt-5",
        "suggest": ["gpt-5-mini", "gpt-5", "gpt-4o-mini", "gpt-4o"],
    },
    "anthropic": {
        "title": "Anthropic (Claude)",
        "url": "https://api.anthropic.com/v1/messages",
        "wire": "anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "key_hint": "https://platform.claude.com/settings/keys",
        "fast": "claude-haiku-4-5",
        "accurate": "claude-opus-4-8",
        "suggest": ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"],
    },
    "deepseek": {
        "title": "DeepSeek",
        "url": "https://api.deepseek.com/chat/completions",
        "wire": "openai",
        "key_env": "DEEPSEEK_API_KEY",
        "key_hint": "https://platform.deepseek.com/api_keys",
        "fast": "deepseek-chat",
        "accurate": "deepseek-reasoner",
        "suggest": ["deepseek-chat", "deepseek-reasoner"],
    },
}

IS_MAC = sys.platform == "darwin"


# ====================== config & setup wizard ======================

def load_config() -> dict | None:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if cfg.get("mode") == "desktop":
        return cfg if IS_MAC else None
    if cfg.get("provider") in PROVIDERS and cfg.get("api_key"):
        return cfg
    return None


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    try:
        os.chmod(CONFIG_PATH, 0o600)  # file holds the API key
    except OSError:
        pass
    print(f"Сохранено: {CONFIG_PATH}")


def _choose(prompt: str, options: list[str]) -> int:
    """Numbered menu; returns the selected index."""
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(f"{prompt} [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print("Введи номер из списка.")


def _ask_api_key(provider: str) -> str:
    p = PROVIDERS[provider]
    env_key = os.environ.get(p["key_env"], "")
    if env_key:
        if input(f"Нашёл ключ в переменной {p['key_env']}. Использовать его? [Y/n]: ").strip().lower() != "n":
            return env_key
    print(f"Ключ можно взять здесь: {p['key_hint']}")
    while True:
        key = input("Вставь API-ключ: ").strip()
        if key:
            return key
        print("Ключ пустой, попробуй ещё раз.")


def setup_wizard() -> dict:
    print("\n=== Настройка ===")

    if IS_MAC:
        idx = _choose(
            "Как получать ответ?",
            [
                "API — ответ печатается в консоль (любая ОС)",
                "Desktop — вопрос отправляется в приложение Claude на Mac",
            ],
        )
        if idx == 1:
            cfg = {"mode": "desktop"}
            save_config(cfg)
            return cfg

    if input("Использовать OpenRouter? [Y/n]: ").strip().lower() != "n":
        provider = "openrouter"
    else:
        others = ["openai", "anthropic", "deepseek"]
        idx = _choose("Выбери провайдера:", [PROVIDERS[k]["title"] for k in others])
        provider = others[idx]

    key = _ask_api_key(provider)
    p = PROVIDERS[provider]
    cfg = {
        "mode": "api",
        "provider": provider,
        "api_key": key,
        "fast_model": p["fast"],
        "accurate_model": p["accurate"],
    }
    save_config(cfg)
    print(f"Модели по умолчанию: быстрая — {p['fast']}, точная — {p['accurate']}")
    print("Сменить их: python3 app.py models")
    return cfg


def models_wizard() -> None:
    cfg = load_config()
    if cfg is None or cfg.get("mode") != "api":
        print("Сначала настрой API-режим: python3 app.py setup")
        return

    p = PROVIDERS[cfg["provider"]]
    print(f"\nПровайдер: {p['title']}")
    print("Доступные варианты (можно вписать любой слаг провайдера):")
    for m in p["suggest"]:
        print(f"  - {m}")

    for label, field in (("быстрого", "fast_model"), ("точного", "accurate_model")):
        current = cfg[field]
        raw = input(f"Модель для {label} ответа [{current}]: ").strip()
        if raw:
            cfg[field] = raw
    save_config(cfg)


# ====================== API client ======================

def _build_request(cfg: dict, model: str, system_prompt: str, question: str,
                   max_tokens: int, stream: bool):
    p = PROVIDERS[cfg["provider"]]
    if p["wire"] == "anthropic":
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": question}],
            "stream": stream,
        }
        headers = {
            "x-api-key": cfg["api_key"],
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    else:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "max_tokens": max_tokens,
            "stream": stream,
        }
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        }
        if cfg["provider"] == "openrouter":
            headers["HTTP-Referer"] = "http://localhost/tilt-handy"
            headers["X-Title"] = "Tilt Handy Assistant"
    return urllib.request.Request(
        p["url"],
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def _extract_text(cfg: dict, data: dict) -> str:
    if PROVIDERS[cfg["provider"]]["wire"] == "anthropic":
        return "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        ).strip()
    return data["choices"][0]["message"]["content"].strip()


def _extract_delta(cfg: dict, data: dict) -> str | None:
    if PROVIDERS[cfg["provider"]]["wire"] == "anthropic":
        if data.get("type") == "content_block_delta" and data["delta"].get("type") == "text_delta":
            return data["delta"]["text"]
        return None
    try:
        return data["choices"][0]["delta"].get("content")
    except (KeyError, IndexError):
        return None


def ask_model(cfg: dict, model: str, question: str, max_tokens: int = 800,
              cancel: threading.Event | None = None) -> str:
    """Full response in one piece (no streaming)."""
    if cancel and cancel.is_set():
        return ""
    request = _build_request(cfg, model, ACCURATE_PROMPT, question, max_tokens, stream=False)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
    except Exception as exc:
        return f"Ошибка запроса: {exc}"
    if cancel and cancel.is_set():
        return ""
    try:
        return _extract_text(cfg, data)
    except (KeyError, IndexError, TypeError):
        return f"Неожиданный ответ API: {data}"


def stream_model(cfg: dict, model: str, question: str, max_tokens: int = 400,
                 cancel: threading.Event | None = None) -> None:
    """Print the response as it is generated (SSE streaming)."""
    request = _build_request(cfg, model, FAST_PROMPT, question, max_tokens, stream=True)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            for raw_line in response:
                if cancel and cancel.is_set():
                    break
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = _extract_delta(cfg, json.loads(data))
                except json.JSONDecodeError:
                    continue
                if delta:
                    print(delta, end="", flush=True)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}")
    except Exception as exc:
        print(f"Ошибка запроса: {exc}")
    print()


def answer_question(cfg: dict, question: str, cancel: threading.Event) -> None:
    # Accurate model runs in the background while the fast one streams.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        accurate = executor.submit(
            ask_model, cfg, cfg["accurate_model"], question, cancel=cancel
        )
        print(f"\n[БЫСТРЫЙ — {cfg['fast_model']}]")
        stream_model(cfg, cfg["fast_model"], question, cancel=cancel)
        if cancel.is_set():
            print("\n[ПРЕРВАНО — новый запрос]")
            return
        print(f"[ТОЧНЫЙ — {cfg['accurate_model']}]")
        result = accurate.result()
        if not cancel.is_set():
            print(result)


# ====================== desktop mode (macOS, Claude.app) ======================

def _osascript(script: str) -> tuple[bool, str]:
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return result.returncode == 0, (result.stdout or result.stderr).strip()


def _frontmost_app() -> str:
    ok, out = _osascript(
        'tell application "System Events" to get name of first application process whose frontmost is true'
    )
    return out if ok else ""


def prepare_claude_window() -> bool:
    """Focus Claude.app BEFORE stopping Handy so its auto-paste lands in the chat."""
    was_running = _osascript('application "Claude" is running')[1] == "true"
    ok, err = _osascript('tell application "Claude" to activate')
    if not ok:
        print(f"Не смог активировать Claude: {err}")
        return False
    if not was_running:
        time.sleep(3.0)

    deadline = time.time() + 5.0
    front = ""
    while time.time() < deadline:
        front = _frontmost_app()
        if front == "Claude":
            break
        time.sleep(0.1)
    else:
        print(f"Claude так и не стал активным окном (frontmost: {front!r}).")
        return False

    time.sleep(0.3)
    if NEW_CHAT_PER_QUERY:
        ok, err = _osascript('tell application "System Events" to keystroke "n" using command down')
        if not ok:
            print(f"osascript ошибка (Cmd+N): {err}")
            print("Проверь: System Settings → Privacy & Security → Accessibility → добавь Terminal")
            return False
        time.sleep(0.5)
    print("Claude в фокусе, готов к вставке.")
    return True


def submit_question_to_claude() -> None:
    """Press Enter — the text is already in the input, pasted by Handy."""
    if _frontmost_app() != "Claude":
        _osascript('tell application "Claude" to activate')
        time.sleep(0.5)
        if _frontmost_app() != "Claude":
            print("Claude не в фокусе — не отправляю, чтобы не печатать в чужое окно.")
            return
    ok, err = _osascript('tell application "System Events" to key code 36')
    if not ok:
        print(f"osascript ошибка (Enter): {err}")
        return
    print("Отправил запрос в Claude.")


# ====================== Handy & clipboard ======================

def find_handy_binary() -> str | None:
    candidates = []
    if IS_MAC:
        candidates.append(pathlib.Path("/Applications/Handy.app/Contents/MacOS/Handy"))
    elif sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            candidates.append(pathlib.Path(local) / "Programs" / "Handy" / "handy.exe")
    for c in candidates:
        if c.exists():
            return str(c)
    return shutil.which("handy")


def toggle_handy_transcription(handy_binary: str) -> bool:
    try:
        subprocess.Popen(
            [handy_binary, "--toggle-transcription"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        print(f"Не смог запустить Handy: {exc}")
        return False
    return True


def get_clipboard_text() -> str:
    if IS_MAC:
        cmd = ["pbpaste"]
    elif sys.platform == "win32":
        cmd = ["powershell", "-noprofile", "-command", "Get-Clipboard"]
    else:
        cmd = ["xclip", "-selection", "clipboard", "-o"]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def wait_for_clipboard_update(previous_text: str, timeout_sec: float) -> str | None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        text = get_clipboard_text()
        if text and text != previous_text:
            return text
        time.sleep(0.2)
    return None


def process_transcription(cfg: dict, handy_binary: str, previous_clipboard: str,
                          cancel: threading.Event) -> None:
    desktop = cfg["mode"] == "desktop"

    # Desktop mode: focus Claude first so Handy's auto-paste lands in it.
    claude_ready = prepare_claude_window() if desktop else False

    if not toggle_handy_transcription(handy_binary):
        return
    print("Handy: запись остановлена")

    print("Жду транскрибацию в буфере обмена...")
    question = wait_for_clipboard_update(previous_clipboard, CLIPBOARD_TIMEOUT_SEC)
    if cancel.is_set():
        return
    if question is None:
        print("Не дождался нового текста в буфере обмена.")
        return
    print(f"\n--- Вопрос ---\n{question}")

    if desktop:
        if not claude_ready:
            print("Claude не готов — текст в буфере обмена, вставь вручную (Cmd+V).")
            return
        time.sleep(0.6)  # let Handy finish pasting into Claude's input
        if not cancel.is_set():
            submit_question_to_claude()
    else:
        answer_question(cfg, question, cancel)


# ====================== tilt detection ======================

def ensure_model() -> pathlib.Path:
    model_path = pathlib.Path(__file__).resolve().with_name(MODEL_NAME)
    if model_path.exists():
        return model_path
    print(f"Скачиваю модель MediaPipe: {MODEL_NAME}")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Не удалось скачать модель {MODEL_NAME}.\n"
            f"Скачай вручную:\n{MODEL_URL}\nи положи рядом со скриптом."
        ) from exc
    return model_path


def create_face_landmarker(mp, model_path: pathlib.Path):
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def run(cfg: dict) -> None:
    import cv2
    import mediapipe as mp

    if not hasattr(mp, "tasks"):
        print("Установлен старый mediapipe без mp.tasks. Переустанови: pip install -U mediapipe")
        return

    handy_binary = find_handy_binary()
    if handy_binary is None:
        print("Не нашёл Handy (приложение или команду handy в PATH).")
        return

    face_landmarker = create_face_landmarker(mp, ensure_model())
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Не открылась камера {CAM_INDEX}.")
        face_landmarker.close()
        return

    baseline = None
    armed = False
    last_trigger_time = 0.0
    angle_hist = collections.deque(maxlen=3)
    start_time = time.monotonic()
    handy_recording = False
    _answer_cancel = threading.Event()
    _answer_cancel.set()

    mode_desc = "Claude desktop" if cfg["mode"] == "desktop" else \
        f"{PROVIDERS[cfg['provider']]['title']} ({cfg['fast_model']} + {cfg['accurate_model']})"
    print(f"Режим: {mode_desc}")
    print("Наклон головы вбок = старт/стоп записи. Ctrl+C или 'q' — выход.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            now = time.time()

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                h, w = frame.shape[:2]
                lx, ly = lm[LEFT_EYE].x * w, lm[LEFT_EYE].y * h
                rx, ry = lm[RIGHT_EYE].x * w, lm[RIGHT_EYE].y * h
                # Eye-line roll angle, normalized to [-90, 90].
                angle = math.degrees(math.atan2(ry - ly, rx - lx))
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180
                angle_hist.append(angle)
                angle = sum(angle_hist) / len(angle_hist)

                if baseline is None:
                    baseline = angle
                tilt = abs(angle - baseline)

                if (
                    not armed
                    and tilt > TILT_ON_DEG
                    and (now - last_trigger_time) > COOLDOWN_SEC
                ):
                    armed = True
                elif armed and tilt < TILT_OFF_DEG:
                    armed = False
                    last_trigger_time = now
                    print(f"[{time.strftime('%H:%M:%S')}] НАКЛОН")
                    if not handy_recording:
                        if toggle_handy_transcription(handy_binary):
                            handy_recording = True
                            print("Handy: запись запущена")
                    else:
                        handy_recording = False
                        previous_clipboard = get_clipboard_text()
                        _answer_cancel.set()  # cancel the previous request
                        _answer_cancel = threading.Event()
                        threading.Thread(
                            target=process_transcription,
                            args=(cfg, handy_binary, previous_clipboard, _answer_cancel),
                            daemon=True,
                        ).start()

                if not armed:
                    baseline = (1 - BASELINE_ALPHA) * baseline + BASELINE_ALPHA * angle

                if SHOW_WINDOW:
                    cv2.line(frame, (int(lx), int(ly)), (int(rx), int(ry)), (0, 255, 0), 2)
                    state = "ARMED" if armed else "watching"
                    cv2.putText(frame, f"{state}  tilt={tilt:.0f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif SHOW_WINDOW:
                cv2.putText(frame, "no face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if SHOW_WINDOW:
                cv2.imshow("tilt detect (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_landmarker.close()
        print("\nОстановлено.")


def main() -> None:
    args = sys.argv[1:]
    if args:
        if args[0] == "setup":
            setup_wizard()
            return
        if args[0] == "models":
            models_wizard()
            return
        print(__doc__)
        return

    cfg = load_config()
    if cfg is None:
        cfg = setup_wizard()
    run(cfg)


if __name__ == "__main__":
    main()
