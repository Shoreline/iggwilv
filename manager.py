import os
import sys
import json
import time
import queue
import pathlib
import tempfile
import threading
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# Ensure clean UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

APP_DIR = pathlib.Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
SETTINGS_FILE = APP_DIR / "local_settings.json"
CONFIG_FILE = APP_DIR / "manager_config.json"

# Machine-local default locations (used only until the user picks paths)
DEFAULT_MODELS_DIR = BASE_DIR / "models"
DEFAULT_ENGINES_DIR = BASE_DIR / "engines"


def load_settings():
    """Read the machine-local settings file (never committed to git)."""
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_settings(settings):
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception as e:
        print("Failed to save settings:", e)
        return False


def get_models_dir():
    value = load_settings().get("models_dir")
    return pathlib.Path(value) if value else DEFAULT_MODELS_DIR


def get_engines_dir():
    value = load_settings().get("engines_dir")
    return pathlib.Path(value) if value else DEFAULT_ENGINES_DIR


def resolve_comfy_exe():
    """Locate Comfy Desktop without hardcoding a machine-specific path.

    Resolution order: COMFY_DESKTOP_EXE env var -> local_settings.json
    {"comfy_exe": "..."} -> common install locations. Returns None when
    nothing is found; the feature then reports as unavailable.
    """
    env_path = os.environ.get("COMFY_DESKTOP_EXE")
    if env_path:
        return pathlib.Path(env_path)
    value = load_settings().get("comfy_exe")
    if value:
        return pathlib.Path(value)
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        for candidate in (
            pathlib.Path(local_appdata) / "Programs" / "ComfyUI" / "Comfy Desktop.exe",
            pathlib.Path(local_appdata) / "Programs" / "comfyui" / "Comfy Desktop.exe",
        ):
            if candidate.exists():
                return candidate
    return None

DEFAULT_SETTINGS = {
    "context": 32768,
    "ngl": 99,
    "threads": 8,
    "threads_batch": 16,
    "batch": 2048,
    "ubatch": 512,
    "temp": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.05,
    "flash_attn": True,
    "jinja": True,
    "mlock": True,
    "enable_mcp": True,
    "cache_k": "q8_0",
    "cache_v": "q8_0",
    "vram_reserve": 256,
    "port": 8080,
    "host": "127.0.0.1",
    "extra_args": ""
}

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class ProcessManager:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.logs = []
        self.log_lock = threading.Lock()
        self.status = "stopped" # stopped, starting, running, error
        self.current_model = None
        self.current_engine = None
        self.port = 8080
        self.start_time = None
        self.error_msg = ""

    def append_log(self, text):
        with self.log_lock:
            self.logs.append(text)
            if len(self.logs) > 2500:
                self.logs.pop(0)

    def reader_thread(self, proc):
        try:
            for line in iter(proc.stdout.readline, ''):
                if line:
                    clean_line = line.rstrip()
                    self.append_log(clean_line)
                    if "HTTP server listening" in clean_line or "model loaded" in clean_line:
                        with self.lock:
                            if self.status == "starting":
                                self.status = "running"
                else:
                    break
        except Exception as e:
            self.append_log(f"[Manager Exception reading output] {e}")
        finally:
            proc.stdout.close()
            ret = proc.wait()
            with self.lock:
                if self.process == proc:
                    self.process = None
                    if ret != 0 and self.status == "starting":
                        self.status = "error"
                        self.error_msg = f"Process exited with code {ret}"
                    else:
                        self.status = "stopped"
                    self.append_log(f"[Manager] Process terminated (exit code {ret}).")

    # Map the generic web-UI KV-cache choices onto Strata's --kv values.
    STRATA_KV_MAP = {
        "q8_0": "int8",
        "q4_0": "q4_0",
        "f16": "fp16",
        "fp16": "fp16",
        "int8": "int8",
        "default": None,
    }

    def _apply_strata_overrides(self, cfg, params):
        """Map the web UI's generic params onto Strata's engine args so that the
        context / KV cache / VRAM reserve shown in the UI actually take effect."""
        args = list(cfg.get("args", []))

        def set_flag(flag, value):
            if value is None:
                return
            if flag in args:
                i = args.index(flag)
                if i + 1 < len(args):
                    args[i + 1] = str(value)
                else:
                    args.append(str(value))
            else:
                args.extend([flag, str(value)])

        # context -> --max-context
        try:
            set_flag("--max-context", int(params.get("context")))
        except (TypeError, ValueError):
            pass

        # KV cache -> --kv (q8_0 -> int8, f16 -> fp16; "default" leaves it as-is)
        kv_src = str(params.get("cache_k") or params.get("cache_v") or "").lower()
        kv_val = self.STRATA_KV_MAP.get(kv_src)
        set_flag("--kv", kv_val)
        self.append_log(
            f"[Manager] Strata: context={params.get('context')} | --kv={kv_val or '(unchanged)'}"
        )

        # VRAM reserve -> --vram-reserve-mib
        try:
            vram = int(params.get("vram_reserve"))
            set_flag("--vram-reserve-mib", vram)
            self.append_log(f"[Manager] Strata: --vram-reserve-mib={vram}")
        except (TypeError, ValueError):
            pass

        # MCP toggle -> mcp_servers.web.disabled
        enable_mcp = params.get("enable_mcp", True)
        if "mcp_servers" in cfg and isinstance(cfg["mcp_servers"], dict) and "web" in cfg["mcp_servers"]:
            cfg["mcp_servers"]["web"]["disabled"] = not bool(enable_mcp)
            self.append_log(f"[Manager] MCP Tools for Strata: {'ENABLED' if enable_mcp else 'DISABLED'}")

        cfg["args"] = args

    def start(self, model_name, engine_name, params):
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "Server is already running."

            self.status = "starting"
            self.error_msg = ""
            self.current_model = model_name
            self.current_engine = engine_name
            self.port = int(params.get("port", 8080))
            self.start_time = time.time()
            self.logs.clear()

            engine_path = get_engines_dir() / engine_name

            # Special launcher for Strata
            if engine_name.lower() == "strata":
                strata_py = engine_path / ".venv" / "Scripts" / "python.exe"
                if not strata_py.exists():
                    strata_py = pathlib.Path(sys.executable)
                server_py = engine_path / "serve" / "server.py"
                if "swift" in model_name.lower() or "xxs" in model_name.lower():
                    cfg_json = engine_path / "strata-swift-iq3_xxs.json"
                else:
                    cfg_json = engine_path / "strata-iq3_s.json"

                if not cfg_json.exists():
                    self.status = "error"
                    self.error_msg = f"Strata config not found: {cfg_json}"
                    return False, self.error_msg

                # Build a runtime config = template JSON + the web UI's overrides,
                # so the UI's context / KV / VRAM-reserve actually drive Strata
                # without mutating the hand-tuned template JSON.
                try:
                    with open(cfg_json, "r", encoding="utf-8") as f:
                        raw_cfg = json.load(f)
                    self._apply_strata_overrides(raw_cfg, params)
                    runtime_cfg = pathlib.Path(tempfile.gettempdir()) / f"strata-runtime-{cfg_json.stem}.json"
                    with open(runtime_cfg, "w", encoding="utf-8") as f:
                        json.dump(raw_cfg, f, indent=1, ensure_ascii=False)
                except Exception as e:
                    self.status = "error"
                    self.error_msg = f"Failed to prepare Strata config: {e}"
                    return False, self.error_msg

                cmd = [
                    str(strata_py),
                    str(server_py),
                    "--engine", "strata",
                    "--config", str(runtime_cfg),
                    "--port", str(self.port)
                ]
            else:
                exe_path = engine_path / "llama-server.exe"
                if not exe_path.exists():
                    alt_exe = engine_path / "strata.exe"
                    if alt_exe.exists():
                        exe_path = alt_exe
                    else:
                        self.status = "error"
                        self.error_msg = f"Executable not found in {engine_path}"
                        return False, self.error_msg

                model_file = get_models_dir() / model_name
                if not model_file.exists():
                    self.status = "error"
                    self.error_msg = f"Model file not found: {model_name}"
                    return False, self.error_msg

                # Build command line for llama.cpp / prism
                cmd = [
                    str(exe_path),
                    "-m", str(model_file),
                    "-ngl", str(params.get("ngl", 99)),
                    "-c", str(params.get("context", 16384)),
                    "-b", str(params.get("batch", 512)),
                    "-ub", str(params.get("ubatch", 128)),
                    "--host", str(params.get("host", "127.0.0.1")),
                    "--port", str(self.port)
                ]

                mmproj = params.get("mmproj")
                if mmproj and mmproj != "none":
                    mmproj_file = MODELS_DIR / mmproj
                    if mmproj_file.exists():
                        cmd.extend(["--mmproj", str(mmproj_file)])

                if params.get("flash_attn", True):
                    cmd.extend(["--flash-attn", "on"])
                if params.get("jinja", True):
                    cmd.append("--jinja")
                if params.get("mlock", True):
                    cmd.append("--mlock")

                if params.get("threads"):
                    cmd.extend(["-t", str(params.get("threads"))])
                if params.get("threads_batch"):
                    cmd.extend(["-tb", str(params.get("threads_batch"))])

                if params.get("cache_k") and params.get("cache_k") != "default":
                    cmd.extend(["--cache-type-k", str(params.get("cache_k"))])
                if params.get("cache_v") and params.get("cache_v") != "default":
                    cmd.extend(["--cache-type-v", str(params.get("cache_v"))])

                if params.get("temp") is not None:
                    cmd.extend(["--temp", str(params.get("temp"))])
                if params.get("top_p") is not None:
                    cmd.extend(["--top-p", str(params.get("top_p"))])
                if params.get("top_k") is not None:
                    cmd.extend(["--top-k", str(params.get("top_k"))])
                if params.get("min_p") is not None:
                    cmd.extend(["--min-p", str(params.get("min_p"))])

                webui_cfg = engine_path / "webui-config.json"
                if webui_cfg.exists():
                    cmd.extend(["--webui-config-file", "webui-config.json"])

                extra = params.get("extra_args", "").strip()
                if extra:
                    import shlex
                    try:
                        cmd.extend(shlex.split(extra))
                    except Exception:
                        cmd.extend(extra.split())

            self.append_log(f"[Manager] Spawning command in {engine_path}:")
            self.append_log(" ".join(cmd))

            try:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=str(engine_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1
                )
                t = threading.Thread(target=self.reader_thread, args=(self.process,), daemon=True)
                t.start()
                return True, "Process started successfully."
            except Exception as e:
                self.status = "error"
                self.error_msg = str(e)
                self.append_log(f"[Manager] Failed to start process: {e}")
                return False, str(e)

    def stop(self):
        with self.lock:
            if not self.process or self.process.poll() is None:
                if self.process:
                    self.append_log("[Manager] Stopping process gracefully...")
                    try:
                        self.process.terminate()
                        # Wait up to 3 seconds
                        for _ in range(15):
                            if self.process.poll() is not None:
                                break
                            time.sleep(0.2)
                        if self.process.poll() is None:
                            self.append_log("[Manager] Force killing process...")
                            self.process.kill()
                    except Exception as e:
                        self.append_log(f"[Manager] Error stopping process: {e}")
                self.process = None
                self.status = "stopped"
                self.append_log("[Manager] Server stopped.")
                return True, "Server stopped."
            return True, "Server already stopped."

    def get_status(self):
        with self.lock:
            uptime = int(time.time() - self.start_time) if self.start_time and self.status == "running" else 0
            # Double check if process died
            if self.process and self.process.poll() is not None and self.status in ("running", "starting"):
                self.status = "stopped"
            return {
                "status": self.status,
                "model": self.current_model,
                "engine": self.current_engine,
                "port": self.port,
                "uptime": uptime,
                "error": self.error_msg,
                "url": f"http://127.0.0.1:{self.port}" if self.status == "running" else None
            }

manager = ProcessManager()

def scan_files():
    models = []
    mmproj_list = []
    models_dir = get_models_dir()
    if models_dir.exists():
        for f in sorted(models_dir.rglob("*.gguf")):
            # Ignore cache folder
            if ".cache" in f.parts:
                continue
            
            # Relative path from the models directory
            rel_name = str(f.relative_to(models_dir)).replace("\\", "/")
            base_name = f.name
            size_gb = round(f.stat().st_size / (1024**3), 2)
            
            # Check for mmproj
            if "mmproj" in base_name.lower():
                mmproj_list.append({"name": rel_name, "size_gb": size_gb})
            # Ignore mtp draft heads as main models
            elif "mtp" in base_name.lower():
                continue
            # Ignore second/third shards of split models
            elif "-of-0000" in base_name and not base_name.endswith("00001-of-00002.gguf") and not "-00001-of-" in base_name:
                continue
            else:
                models.append({"name": rel_name, "size_gb": size_gb})

    engines = []
    engines_dir = get_engines_dir()
    if engines_dir.exists():
        for d in sorted(engines_dir.iterdir()):
            if d.is_dir() and not d.name.endswith("-data"):
                has_server = (d / "llama-server.exe").exists() or (d / "strata.exe").exists() or (d / "START-HERE.bat").exists()
                if has_server:
                    engines.append({
                        "name": d.name,
                        "has_server": has_server
                    })

    return models, mmproj_list, engines

def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "models": {},
        "last_selected_model": None
    }

def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print("Failed to save config:", e)
        return False

def get_smart_defaults_for_model(model_name, mmproj_list, engines):
    name_lower = model_name.lower()
    res = dict(DEFAULT_SETTINGS)

    # 1. Match Engine
    if "bonsai" in name_lower:
        prism_eng = next((e["name"] for e in engines if "prism" in e["name"].lower()), None)
        if prism_eng:
            res["engine"] = prism_eng
        res["context"] = 65536
        res["temp"] = 1.0
        res["top_p"] = 0.95
        res["top_k"] = 20
        res["min_p"] = 0.0
        res["cache_k"] = "q4_0"
        res["cache_v"] = "q4_0"
    elif "flash-next" in name_lower:
        strata_eng = next((e["name"] for e in engines if "strata" in e["name"].lower()), None)
        if strata_eng:
            res["engine"] = strata_eng
        res["context"] = 65536
        res["temp"] = 0.7
        res["cache_k"] = "q8_0"
        res["cache_v"] = "q8_0"
    elif "35b" in name_lower or "qwen3.6" in name_lower:
        std_eng = next((e["name"] for e in engines if "b10199" in e["name"].lower()), None)
        if not std_eng and engines:
            std_eng = engines[0]["name"]
        res["engine"] = std_eng
        res["context"] = 32768
        res["extra_args"] = "--n-cpu-moe 999"
        res["cache_k"] = "q8_0"
        res["cache_v"] = "q8_0"
        res["temp"] = 0.8
    elif "27b" in name_lower:
        std_eng = next((e["name"] for e in engines if "b10199" in e["name"].lower()), None)
        if not std_eng and engines:
            std_eng = engines[0]["name"]
        res["engine"] = std_eng
        res["context"] = 32768
        res["cache_k"] = "default"
        res["cache_v"] = "default"
        res["temp"] = 0.7
        res["top_p"] = 0.95
    else:
        std_eng = next((e["name"] for e in engines if "b10199" in e["name"].lower()), None)
        if not std_eng and engines:
            std_eng = engines[0]["name"]
        res["engine"] = std_eng

    # 2. Match MMPROJ
    matched_mm = "none"
    if "bonsai" in name_lower:
        mm = next((m["name"] for m in mmproj_list if "bonsai" in m["name"].lower()), None)
        if mm: matched_mm = mm
    elif "flash-next" in name_lower:
        if "swift" in name_lower:
            mm = next((m["name"] for m in mmproj_list if "swift" in m["name"].lower()), None)
        else:
            mm = next((m["name"] for m in mmproj_list if "flash-next" in m["name"].lower() and "swift" not in m["name"].lower()), None)
        if not mm:
            mm = next((m["name"] for m in mmproj_list if "flash-next" in m["name"].lower()), None)
        if mm: matched_mm = mm
    elif "35b" in name_lower or "qwen3.6" in name_lower:
        mm = next((m["name"] for m in mmproj_list if "35b" in m["name"].lower() or "qwen3.6" in m["name"].lower()), None)
        if mm: matched_mm = mm
        res["extra_args"] = "--n-cpu-moe 999"

    res["mmproj"] = matched_mm
    return res

class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence standard HTTP access logging to keep console clean
        return

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html_path = pathlib.Path(__file__).resolve().parent / "index.html"
            if html_path.exists():
                with open(html_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.end_headers()
            return

        if self.path == "/api/data" or self.path.startswith("/api/data?"):
            models, mmproj_list, engines = scan_files()
            cfg = load_config()

            # Generate pure recommended presets for each model
            recommended_presets = {}
            for m in models:
                m_name = m["name"]
                recommended_presets[m_name] = get_smart_defaults_for_model(m_name, mmproj_list, engines)

            # Ensure all discovered models have config entries
            model_configs = cfg.get("models", {})
            for m in models:
                m_name = m["name"]
                if m_name not in model_configs:
                    model_configs[m_name] = dict(recommended_presets[m_name])

            # Auto default engine if missing
            for m_name, c in model_configs.items():
                if not c.get("engine") and engines:
                    rec_eng = recommended_presets.get(m_name, {}).get("engine")
                    c["engine"] = rec_eng if rec_eng else engines[0]["name"]

            comfy = resolve_comfy_exe()
            models_dir = get_models_dir()
            engines_dir = get_engines_dir()
            data = {
                "models": models,
                "mmproj_list": mmproj_list,
                "engines": engines,
                "configs": model_configs,
                "recommended": recommended_presets,
                "last_selected_model": cfg.get("last_selected_model") or (models[0]["name"] if models else None),
                "status": manager.get_status(),
                "comfy_installed": bool(comfy and comfy.exists()),
                "configured": models_dir.exists() and engines_dir.exists(),
                "models_dir": str(models_dir),
                "engines_dir": str(engines_dir),
                "comfy_exe": str(comfy) if comfy else ""
            }
            self.send_json(data)
            return

        if self.path.startswith("/api/logs"):
            # Support offset
            offset = 0
            if "offset=" in self.path:
                try:
                    offset = int(self.path.split("offset=")[1].split("&")[0])
                except Exception:
                    pass
            with manager.log_lock:
                total_len = len(manager.logs)
                lines = manager.logs[offset:] if offset < total_len else []
                self.send_json({
                    "lines": lines,
                    "total": total_len,
                    "status": manager.get_status()
                })
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            req = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            req = {}

        if self.path == "/api/save_config":
            cfg = load_config()
            model_name = req.get("model")
            model_params = req.get("params", {})
            if model_name:
                if "models" not in cfg:
                    cfg["models"] = {}
                cfg["models"][model_name] = model_params
                cfg["last_selected_model"] = model_name
                save_config(cfg)
                self.send_json({"ok": True})
            else:
                self.send_json({"ok": False, "error": "Missing model name"}, 400)
            return

        if self.path == "/api/start":
            model_name = req.get("model")
            engine_name = req.get("engine")
            params = req.get("params", {})

            # Save as last selected
            cfg = load_config()
            cfg["last_selected_model"] = model_name
            if "models" not in cfg: cfg["models"] = {}
            cfg["models"][model_name] = params
            save_config(cfg)

            ok, msg = manager.start(model_name, engine_name, params)
            if ok:
                if req.get("launch_comfy", False):
                    self.launch_comfy_internal()
                self.send_json({"ok": True, "message": msg})
            else:
                self.send_json({"ok": False, "error": msg}, 400)
            return

        if self.path == "/api/stop":
            ok, msg = manager.stop()
            self.send_json({"ok": ok, "message": msg})
            return

        if self.path == "/api/launch_comfy":
            ok, msg = self.launch_comfy_internal()
            self.send_json({"ok": ok, "message": msg})
            return

        if self.path == "/api/set_paths":
            models_dir = (req.get("models_dir") or "").strip()
            engines_dir = (req.get("engines_dir") or "").strip()
            comfy_exe = (req.get("comfy_exe") or "").strip()

            errors = []
            if not models_dir:
                errors.append("models_dir is required")
            elif not pathlib.Path(models_dir).is_dir():
                errors.append(f"Models directory not found: {models_dir}")
            if not engines_dir:
                errors.append("engines_dir is required")
            elif not pathlib.Path(engines_dir).is_dir():
                errors.append(f"Engines directory not found: {engines_dir}")
            if comfy_exe and not pathlib.Path(comfy_exe).exists():
                errors.append(f"Comfy executable not found: {comfy_exe}")

            if errors:
                self.send_json({"ok": False, "error": "; ".join(errors)}, 400)
                return

            settings = load_settings()
            settings["models_dir"] = str(pathlib.Path(models_dir).resolve())
            settings["engines_dir"] = str(pathlib.Path(engines_dir).resolve())
            if comfy_exe:
                settings["comfy_exe"] = str(pathlib.Path(comfy_exe).resolve())
            elif "comfy_exe" in settings:
                del settings["comfy_exe"]

            if not save_settings(settings):
                self.send_json({"ok": False, "error": f"Could not write {SETTINGS_FILE}"}, 500)
                return

            self.send_json({"ok": True, "message": "Paths saved.", "models_dir": settings["models_dir"], "engines_dir": settings["engines_dir"]})
            return

        self.send_response(404)
        self.end_headers()

    def launch_comfy_internal(self):
        comfy = resolve_comfy_exe()
        if not comfy or not comfy.exists():
            return False, "Comfy Desktop not found. Set it in the path settings (or the COMFY_DESKTOP_EXE environment variable)."
        try:
            subprocess.Popen([str(comfy)], shell=True)
            return True, "Comfy Desktop launched."
        except Exception as e:
            return False, str(e)

def main():
    port = 8765
    server_address = ("127.0.0.1", port)
    httpd = ThreadedHTTPServer(server_address, RequestHandler)
    url = f"http://127.0.0.1:{port}"
    print("=" * 60)
    print("       * LocalLLM Studio - Web Manager is Online! *       ")
    print("=" * 60)
    print(f" Web Dashboard : {url}")
    print(f" Models  Folder: {get_models_dir()}")
    print(f" Engines Folder: {get_engines_dir()}")
    if not (get_models_dir().exists() and get_engines_dir().exists()):
        print(" [Setup needed] No valid models/engines folder yet — the web page")
        print("               will ask you to pick them; saved to local_settings.json.")
    print(" Press Ctrl+C in this console to shutdown the manager.")
    print("=" * 60)

    # Automatically open browser
    import webbrowser
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Manager] Shutting down...")
        manager.stop()
        httpd.server_close()
        print("[Manager] Bye!")

if __name__ == "__main__":
    main()
