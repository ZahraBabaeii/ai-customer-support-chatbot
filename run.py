"""Start the backend and the chat UI together:  python run.py"""

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from app.config import get_settings

ROOT = Path(__file__).resolve().parent
UI_PORT = 8501


def main() -> int:
    settings = get_settings()
    api_url = f"http://{settings.api_host}:{settings.api_port}"

    backend = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--host", settings.api_host, "--port", str(settings.api_port),
        ],
        cwd=ROOT,
    )
    processes = [backend]
    try:
        print(f"Starting backend at {api_url} (first run downloads the embedding model)...")
        deadline = time.time() + 300
        while True:
            if backend.poll() is not None:
                print("Backend exited during startup. See the log output above.")
                return 1
            try:
                if httpx.get(f"{api_url}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                print("Backend did not become ready within 5 minutes.")
                return 1
            time.sleep(1)

        frontend = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py",
                "--server.port", str(UI_PORT),
            ],
            cwd=ROOT,
            env={**os.environ, "BACKEND_URL": api_url},
        )
        processes.append(frontend)
        print(f"\nChat UI:  http://localhost:{UI_PORT}")
        print(f"API docs: {api_url}/docs")
        print("Press Ctrl+C to stop.\n")

        while all(p.poll() is None for p in processes):
            time.sleep(1)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    sys.exit(main())
