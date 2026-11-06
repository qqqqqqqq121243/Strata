"""serve/reverse_ssh.py - make this PC's Strata server reachable at a remote address (a reverse SSH tunnel).

`ssh -R 8088:127.0.0.1:8080 root@host` makes the remote machine listen on 8088 and forward every connection back
here.  serve/server.py starts it at boot when the JSON below has "enabled": true, and it runs on its own too:

    python serve/reverse_ssh.py --config strata-reverse-ssh.json

Windows' OpenSSH has no sshpass, so the password reaches ssh through an askpass helper (SSH_ASKPASS with
SSH_ASKPASS_REQUIRE=force) that prints it from an environment variable - it is not written to disk, only the config's
password is.  `ssh -N` says nothing when the forward is in place, so a run that is still alive after a grace period
is reported up (ExitOnForwardFailure makes ssh exit instead when the remote port cannot be bound), and one that exits
is reported failed with its last stderr line.  The supervisor restarts it with a growing delay, which is also what
brings the tunnel back after a network drop; closing Strata ends it (a Windows job object, serve/winjob).
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))               # run as a script as well as a module
from serve.winjob import contain  # noqa: E402

DEFAULT_CONFIG = ROOT / "strata-reverse-ssh.json"

# the askpass program: ssh runs it and reads the password from its stdout (see the module docstring)
_ASKPASS_PY = 'import os, sys\nsys.stdout.write(os.environ.get("STRATA_SSH_PASSWORD", ""))\n'


def _num(cfg: dict, key: str, default, cast):
    """A config number, or (default, "what is wrong with it"): a typo shows in the badge instead of stopping Strata."""
    value = cfg.get(key)
    if value in (None, ""):
        return default, ""
    try:
        return cast(value), ""
    except (TypeError, ValueError):
        return default, f"{key} must be a number, not {value!r}"


class ReverseSSH:
    """One `ssh -R` process, kept alive.  `status()` is what the web app's header badge shows."""

    def __init__(self, cfg: dict | None = None, log=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled"))
        self.remote = str(cfg.get("remote") or "")              # "root@134.175.144.49"
        self.ssh = str(cfg.get("ssh") or "ssh")                 # the client, if it is not on PATH
        self.ssh_args = [str(x) for x in (cfg.get("ssh_args") or [])]
        self.port, e1 = _num(cfg, "port", 8088, int)            # the remote machine listens here
        self.local_port, e2 = _num(cfg, "local_port", 0, int)   # ... and forwards back to 127.0.0.1:<this>; 0 = the
        self.grace_s, e3 = _num(cfg, "grace_s", 3.0, float)     #     server's own port, which fills it in at boot
        self.retry_s, e4 = _num(cfg, "retry_s", 5.0, float)
        self.retry_max_s, e5 = _num(cfg, "retry_max_s", 60.0, float)
        self._password = str(cfg.get("password") or "")
        # a broken file's message (from_config) or a bad number: reported as failed, not raised
        self._config_error = "; ".join(str(x) for x in (cfg.get("error"), e1, e2, e3, e4, e5) if x)
        self._log = log or (lambda msg: print(f"[strata] reverse SSH: {msg}", flush=True))
        self._lock = threading.Lock()
        self._state = "off"                                     # off | starting | up | failed
        self._error = ""
        self._since = None
        self._restarts = 0
        self._proc = None
        self._thread = None
        self._stopped = threading.Event()
        self._askpass = None

    # ---------------------------------------------------------------- config
    @classmethod
    def from_config(cls, src, log=None) -> "ReverseSSH":
        """src: a JSON path - a missing file means off, the usual case for an install without a tunnel - or a dict."""
        if isinstance(src, dict):
            return cls(src, log=log)
        if not src:
            return cls({}, log=log)
        path = Path(src)
        try:
            cfg = json.loads(path.read_text(encoding="utf-8-sig"))      # Notepad adds a BOM
        except FileNotFoundError:
            return cls({}, log=log)
        except (OSError, ValueError) as e:
            return cls({"enabled": True, "error": f"{path}: {e}"}, log=log)
        return cls(cfg, log=log)

    def local(self) -> str:
        return f"127.0.0.1:{self.local_port or 8080}"

    def command(self) -> list[str]:
        """The `ssh` argv: no command, one remote forward, and the checks that make a dead or half-open tunnel say so.
        ExitOnForwardFailure: exit instead of running on with the remote port unbound (taken, or not permitted).
        ServerAlive*: notice a peer that went away.  NumberOfPasswordPrompts=1: the askpass helper always answers the
        same password, so a retry only risks the server's lockout."""
        return [self.ssh, "-N", "-T",
                "-o", "ExitOnForwardFailure=yes",
                "-o", "ServerAliveInterval=20",
                "-o", "ServerAliveCountMax=3",
                "-o", "NumberOfPasswordPrompts=1",
                "-o", "StrictHostKeyChecking=accept-new",
                "-R", f"{self.port}:{self.local()}",
                *self.ssh_args, self.remote]

    # ---------------------------------------------------------------- status
    def status(self) -> dict:
        with self._lock:
            return {"enabled": self.enabled, "state": self._state, "error": self._error,
                    "remote": self.remote, "port": self.port, "local": self.local(),
                    "restarts": self._restarts, "since": self._since,
                    "up_s": round(time.time() - self._since, 1) if self._state == "up" and self._since else None,
                    "command": " ".join(self.command())}

    # ---------------------------------------------------------------- start / stop
    def start(self) -> None:
        """Return at once; the tunnel is supervised on its own thread.  Called once, by the server or by main()."""
        if not self.enabled or self._thread is not None:
            return
        if self._config_error:
            self._fail(self._config_error)
            return
        if not self.remote or not self._password:
            self._fail('the config needs "remote" (user@host) and "password"')
            return
        self._thread = threading.Thread(target=self._supervise, name="reverse-ssh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """End the ssh process and the supervisor; safe to call twice (the server and atexit both do)."""
        self._stopped.set()
        if self._proc is not None:
            self._kill(self._proc)
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=5)
        with self._lock:
            self._state, self._since = "off", None

    def _fail(self, reason: str) -> None:
        with self._lock:
            self._state, self._error = "failed", reason
        self._log(f"failed: {reason}")

    def _supervise(self) -> None:
        delay = self.retry_s
        while not self._stopped.is_set():
            with self._lock:
                self._state, self._error, self._since = "starting", "", None
            proc, errs = self._spawn()
            if proc is None:
                with self._lock:
                    self._restarts += 1
                self._fail(errs[0] if errs else "ssh could not be started")
                if self._stopped.wait(delay):
                    break
                delay = min(self.retry_max_s, delay * 2)
                continue
            with self._lock:
                self._proc = proc
            t0, up = time.monotonic(), False
            while not self._stopped.is_set() and proc.poll() is None:
                if not up and time.monotonic() - t0 >= self.grace_s:
                    up, delay = True, self.retry_s
                    with self._lock:
                        self._state, self._since = "up", time.time()
                    self._log(f"up: {self.remote}:{self.port} -> {self.local()}")
                time.sleep(0.25)
            if self._stopped.is_set():
                self._kill(proc)
                break
            reason = next((l.strip() for l in reversed(errs) if l.strip()), "") or \
                f"ssh exited with code {proc.poll()}"
            with self._lock:
                self._state, self._error, self._since, self._proc = "failed", reason, None, None
                self._restarts += 1
            self._log(f"failed: {reason}")
            if self._stopped.wait(delay):
                break
            delay = min(self.retry_max_s, delay * 2)        # a longer outage: back off, up to retry_max_s
        with self._lock:
            self._state, self._since = "off", None

    # ---------------------------------------------------------------- the ssh process
    def _spawn(self):
        """Start one `ssh -R`; (proc, last stderr lines), or (None, [the reason]) if it could not start."""
        env = dict(os.environ)
        env["SSH_ASKPASS"] = self._askpass_helper()
        env["SSH_ASKPASS_REQUIRE"] = "force"        # use it even with a console attached (Windows included)
        env["STRATA_SSH_PASSWORD"] = self._password
        try:
            proc = subprocess.Popen(self.command(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, env=env, text=True, encoding="utf-8", errors="replace")
        except OSError as e:
            return None, [f"{self.ssh}: {e}"]
        contain(proc)                               # ends with this process, however it ends
        errs: deque[str] = deque(maxlen=20)
        reader = threading.Thread(target=lambda: [errs.append(l) for l in proc.stderr], daemon=True)
        reader.start()
        return proc, errs

    @staticmethod
    def _kill(proc) -> None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        except OSError:
            pass

    def _askpass_helper(self) -> str:
        """Write the askpass program once (a .cmd on Windows: ssh starts a path, it cannot pass it arguments)."""
        if self._askpass:
            return self._askpass
        d = Path(tempfile.mkdtemp(prefix="strata-reverse-ssh-"))
        py = d / "askpass.py"
        py.write_text(_ASKPASS_PY, encoding="utf-8")
        python = sys.executable or "python"
        if os.name == "nt":
            # cmd.exe reads a batch file in its own codepage, which is also the one text writes use on Windows
            helper = d / "askpass.cmd"
            helper.write_text(f'@echo off\r\n"{python}" "{py}"\r\n')
        else:
            helper = d / "askpass.sh"
            helper.write_text(f'#!/bin/sh\nexec "{python}" "{py}"\n', encoding="utf-8")
            helper.chmod(0o700)
        self._askpass = str(helper)
        return self._askpass


def main() -> int:
    ap = argparse.ArgumentParser(description="A reverse SSH tunnel to this PC's Strata server "
                                             "(see the module docstring for the config's keys)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help=f"the JSON to read (default: {DEFAULT_CONFIG.name})")
    a = ap.parse_args()
    tunnel = ReverseSSH.from_config(a.config)
    if not tunnel.enabled:
        print(f'[strata] reverse SSH is off: {a.config} does not have "enabled": true', flush=True)
        return 1
    tunnel.start()
    atexit.register(tunnel.stop)
    print(f"[strata] reverse SSH: {tunnel.remote}:{tunnel.port} -> {tunnel.local()} (Ctrl+C to stop)", flush=True)
    last = None
    try:
        while True:
            time.sleep(1)
            s = tunnel.status()
            if (s["state"], s["error"]) != last:
                last = (s["state"], s["error"])
                print(f"[strata] reverse SSH: {s['state']}" + (f" - {s['error']}" if s["error"] else ""), flush=True)
    except KeyboardInterrupt:
        print("\n[strata] stopping the tunnel ...", flush=True)
        tunnel.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
