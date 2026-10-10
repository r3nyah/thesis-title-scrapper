"""One ask() function for two backends: 'ollama' (local) or 'gemini' (API)."""
import json, os, re, shutil, subprocess, time

LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen2.5:1.5b")        # lightweight (~1GB); try qwen2.5:3b / 7b if your PC can
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # override via env var if the model name changes
_ready = False


def _is_up():
    import ollama
    try:
        ollama.list()
        return True
    except Exception:
        return False


def ensure_ollama():
    """Make sure Ollama is running (starts it if installed) and the model is downloaded."""
    global _ready
    if _ready:
        return
    import ollama
    if not _is_up():
        if not shutil.which("ollama"):
            raise SystemExit("Ollama is not installed. Install it from https://ollama.com/download, "
                             "then close and reopen your terminal.")
        print("Starting Ollama...")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=flags)
        for _ in range(30):
            time.sleep(1)
            if _is_up():
                break
        else:
            raise SystemExit("Ollama failed to start. Open the Ollama app manually and try again.")
    print(f"Making sure model {LOCAL_MODEL} is available (downloaded once)...")
    last = None
    for p in ollama.pull(LOCAL_MODEL, stream=True):
        if p.total and p.completed:
            print(f"\r  {p.status}: {p.completed / p.total:.0%}", end="", flush=True)
        elif p.status != last:
            print(f"\n  {p.status}")
            last = p.status
    print()
    _ready = True


def pull_local_model():
    ensure_ollama()


def ask(prompt, backend="ollama", as_json=False):
    if backend == "gemini":
        from google import genai  # reads GEMINI_API_KEY from the environment
        cfg = {"response_mime_type": "application/json"} if as_json else None
        text = genai.Client().models.generate_content(model=GEMINI_MODEL, contents=prompt, config=cfg).text
    else:
        import ollama
        ensure_ollama()
        r = ollama.chat(model=LOCAL_MODEL, messages=[{"role": "user", "content": prompt}],
                        format="json" if as_json else None)
        text = r["message"]["content"]
    return json.loads(re.sub(r"```json|```", "", text).strip()) if as_json else text
