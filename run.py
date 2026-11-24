"""
Main entry point for Data Drift Detection System.
Launches the high-performance ASGI server (Uvicorn) with FastAPI backend
and serves the modern HTML/CSS/JS frontend on http://localhost:8000.
"""

import os
import socket
import sys
from pathlib import Path

# Add project root to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# Auto-detect local virtualenv and re-execute if not running inside it
# ──────────────────────────────────────────────────────────────────────────────
venv_dir = PROJECT_ROOT / "venv"
if venv_dir.exists():
    venv_python = (
        venv_dir / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else venv_dir / "bin" / "python"
    )
    if venv_python.exists():
        try:
            curr_exe = Path(sys.executable).resolve()
            target_exe = venv_python.resolve()
            if curr_exe != target_exe:
                import subprocess

                cmd = [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]]
                sys.exit(subprocess.call(cmd))
        except Exception:
            pass

import uvicorn
from src.utils import setup_logger

logger = setup_logger("runner")


def find_available_port(host: str, preferred_port: int, max_attempts: int = 15) -> int:
    """Find an available TCP port starting from preferred_port."""
    for port in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return preferred_port


def main() -> None:
    """Initialize and run the Uvicorn application server."""
    host = os.getenv("HOST", "127.0.0.1")
    requested_port = int(os.getenv("PORT", "8000"))
    port = find_available_port(host, requested_port)
    reload_enabled = os.getenv("RELOAD", "false").lower() in ("true", "1", "yes")

    logger.info("=" * 70)
    logger.info("Starting Data Drift Detection System (Enterprise v2.0)")
    logger.info("Python Interpreter: %s", sys.executable)
    if port != requested_port:
        logger.warning(
            "Port %d unavailable/restricted. Automatically assigned to Port %d",
            requested_port,
            port,
        )
    logger.info("Host: %s | Port: %d | Reload: %s", host, port, reload_enabled)
    logger.info("Local Dashboard URL: http://localhost:%d", port)
    logger.info("Interactive API Docs: http://localhost:%d/docs", port)
    logger.info("Production Cloud URL: https://dddsystem.vercel.app")
    logger.info("=" * 70)

    uvicorn_kwargs = {
        "app": "api.app:app",
        "host": host,
        "port": port,
        "reload": reload_enabled,
        "log_level": "info",
        "access_log": True,
    }

    if reload_enabled:
        uvicorn_kwargs["reload_excludes"] = [
            "*.log",
            "*.json",
            "logs/*",
            "reports/*",
            ".git/*",
            ".pytest_cache/*",
        ]

    try:
        uvicorn.run(**uvicorn_kwargs)
    except KeyboardInterrupt:
        logger.info("Server gracefully stopped by user.")


if __name__ == "__main__":
    main()
