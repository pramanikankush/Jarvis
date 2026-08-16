"""Launcher: picks a free port, opens the browser, runs the server."""
import os
import socket
import threading
import webbrowser

import envfile

envfile.load_env()  # honour PORT / GROQ_* from .env before anything reads them

import uvicorn  # noqa: E402


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    port = int(os.environ.get("PORT", 0)) or free_port()
    url = f"http://127.0.0.1:{port}/"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"\n  Jarvis is running at {url}\n  (Ctrl+C to stop)\n")
    uvicorn.run("server:app", host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()