"""serve/history.py - the full content of every request, kept locally and mirrored to the reverse-SSH host.

Every finished /v1/chat/completions or /v1/messages request - the input exactly as it arrived, the answer, the
thinking, usage and timings (the same record serve/server.py's request monitor builds, #332) - is appended to

    <Strata folder>/.history/requests.jsonl            the local spool (append-only, nothing is ever pruned)
and mirrored over ssh to
    <remote_dir>/requests.jsonl                        on the reverse-SSH host (the Tencent Cloud server)

Config: the same JSON file serve/reverse_ssh.py reads (strata-reverse-ssh.json); its "remote" and "password" are
reused.  Optional keys: "history": false turns this service off, "history_dir" is the remote folder (default
"strata-history"), "ssh" the client program (default "ssh").

The web app's Recent requests card browses the mirror read-only:
    GET /requests                    one summary per conversation: first question, last answer, tokens ...
    GET /requests/conversation?id=   the whole conversation: every message, thinking blocks, per-answer numbers
The conversation is the group of requests that started from the same first question (served at the same path).
The mirror is fetched over ssh and cached for a few seconds; when the cloud does not answer, the local spool
answers instead, and records that are spooled but not yet mirrored are merged in by record id, so an answer is
visible right after it was written.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from serve.winjob import contain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "strata-reverse-ssh.json"

MAX_BYTES = 64 * 1024 * 1024        # at most this many trailing bytes of a log are read/parsed
MAX_PENDING = 32 * 1024 * 1024      # ... and of the not-yet-mirrored spool tail merged into it
SNIPPET_Q = 240                     # the one-line "first question" / "last answer" of the list
SNIPPET_A = 240

# the askpass program: ssh runs it and reads the password from its stdout (see serve/reverse_ssh.py)
_ASKPASS_PY = 'import os, sys\nsys.stdout.write(os.environ.get("STRATA_SSH_PASSWORD", ""))\n'


# ------------------------------------------------------------------------------------------------ records
def records_from_bytes(data: bytes) -> list[dict]:
    """The records in a request log, oldest first.  Malformed or half lines are skipped; a record seen twice
    (mirror and spool during a merge) counts once."""
    out, seen = [], set()
    for line in data.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        rid = rec.get("id")
        if isinstance(rid, str):
            if rid in seen:
                continue
            seen.add(rid)
        out.append(rec)
    out.sort(key=lambda r: (r.get("started_at") or 0, str(r.get("id") or "")))
    return out


def _text_of(content) -> str:
    """A message's text: a string as it is, a parts list as its text parts ("[image]" for images)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif p.get("type") in ("image_url", "image"):
                    parts.append("[image]")
        return "\n".join(parts)
    return "" if content is None else str(content)


def _messages_of(record: dict) -> list:
    """The messages array of a record's input (empty when it cannot be parsed - the input is cut at 256 KB)."""
    raw = record.get("input")
    if not isinstance(raw, str):
        return []
    try:
        req = json.loads(raw)
    except ValueError:
        return []
    msgs = req.get("messages") if isinstance(req, dict) else None
    return msgs if isinstance(msgs, list) else []


def _first_user(msgs: list) -> str | None:
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            text = _text_of(m.get("content")).strip()
            if text:
                return text
    return None


def _usage(record: dict, kind: str):
    """Tokens of one record: the OpenAI or the Anthropic spelling; None when the record has no usage."""
    usage = record.get("usage")
    if not isinstance(usage, dict):
        return None
    keys = ("prompt_tokens", "input_tokens") if kind == "in" else ("completion_tokens", "output_tokens")
    for k in keys:
        v = usage.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    return None


def _tok_s(record: dict):
    timings = record.get("timings")
    v = timings.get("predicted_per_second") if isinstance(timings, dict) else None
    return round(v, 1) if isinstance(v, (int, float)) else None


def _snippet(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def build_conversations(records: list[dict]) -> list[dict]:
    """Group records into conversations: the requests that started from the same first question, newest first.
    A request without a user message (rare) is a conversation of its own."""
    groups: dict[str, dict] = {}
    for rec in records:
        msgs = _messages_of(rec)
        q = _first_user(msgs)
        if q is None:
            gid = "r" + str(rec.get("id") or "x")[:12]
            q = ""
        else:
            gid = hashlib.sha1((str(rec.get("path")) + "\x00" + q).encode("utf-8")).hexdigest()[:12]
        groups.setdefault(gid, {"id": gid, "q": q, "records": []})["records"].append(rec)
    out = []
    for g in groups.values():
        recs = g["records"]
        last = recs[-1]
        g.update({
            "requests": len(recs),
            "model": last.get("model"),
            "path": last.get("path"),
            "started": recs[0].get("started_at"),
            "updated": last.get("finished_at") or last.get("started_at"),
            "a": _snippet(last.get("output") or "", SNIPPET_A),
            "tokens_in": sum(_usage(r, "in") or 0 for r in recs),
            "tokens_out": sum(_usage(r, "out") or 0 for r in recs),
            "tok_s": _tok_s(last),
            "message_count": len(_messages_of(last)) + 1,
        })
        out.append(g)
    out.sort(key=lambda g: g.get("updated") or 0, reverse=True)
    return out


def _summary(g: dict) -> dict:
    return {k: g[k] for k in ("id", "q", "a", "model", "path", "requests", "started", "updated",
                              "tokens_in", "tokens_out", "tok_s", "message_count")} | {"q": _snippet(g["q"], SNIPPET_Q)}


def _detail(g: dict) -> dict:
    """One conversation, ready to render: the last request's input carries every earlier turn, and each earlier
    answer's thinking, tokens and speed are looked up from the record that produced it (aligned from the end)."""
    recs = g["records"]
    last = recs[-1]
    msgs = _messages_of(last)
    ans_at = [i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "assistant"]
    shift = len(recs) - 1 - len(ans_at)                 # normally 0: each record adds one answer
    items = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user")
        if role not in ("user", "assistant", "system", "tool"):
            role = "user"
        item = {"role": role, "text": _text_of(m.get("content"))}
        if role == "assistant":
            k = ans_at.index(i) + shift
            if 0 <= k < len(recs) - 1:                  # the last record's answer is not in its own input
                src = recs[k]
                item |= {"reasoning": src.get("reasoning") or "",
                         "time": src.get("finished_at") or src.get("started_at"),
                         "tokens": _usage(src, "out"), "tok_s": _tok_s(src)}
        items.append(item)
    items.append({"role": "assistant", "text": last.get("output") or "",
                  "reasoning": last.get("reasoning") or "",
                  "time": last.get("finished_at") or last.get("started_at"),
                  "tokens": _usage(last, "out"), "tok_s": _tok_s(last), "done": True})
    return _summary(g) | {"messages": items,
                          "truncated": any(r.get("input_truncated") or r.get("output_truncated") or
                                           r.get("reasoning_truncated") for r in recs)}


def _tail_bytes(data: bytes, limit: int) -> bytes:
    """The last `limit` bytes of a log, starting at a line boundary."""
    if len(data) <= limit:
        return data
    cut = data[-limit:]
    nl = cut.find(b"\n")
    return cut[nl + 1:] if nl >= 0 else cut


# ------------------------------------------------------------------------------------------------ the service
class History:
    """The request log and its mirror.  One instance per server; the web app only reads what this wrote."""

    POLL_S = 20.0          # how often the spool is checked for new bytes when nothing pokes it
    CACHE_S = 15.0         # how long a fetched mirror (and its parsing) is reused
    SSH_TIMEOUT_S = 60.0

    def __init__(self, cfg: dict | None = None, root: Path | str = ROOT, log=None):
        cfg = cfg or {}
        self.root = Path(root)
        self.remote = str(cfg.get("remote") or "")
        self.ssh = str(cfg.get("ssh") or "ssh")
        self.password = str(cfg.get("password") or "")
        self.remote_dir = str(cfg.get("history_dir") or "strata-history")
        self._config_error = str(cfg.get("error") or "")
        self.enabled = bool(self.remote and self.password) and cfg.get("history") is not False \
            and not self._config_error
        try:
            self.interval = max(5.0, float(cfg.get("history_interval") or self.POLL_S))
        except (TypeError, ValueError):
            self.interval = self.POLL_S
        self.spool = self.root / ".history" / "requests.jsonl"
        self.state_path = self.root / ".history" / "sync.json"
        self._log = log or (lambda msg: print(f"[strata] request history: {msg}", flush=True))
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._askpass: str | None = None
        self._synced: int | None = None                 # bytes mirrored (loaded from sync.json lazily)
        self._since: float | None = None
        self._error = self._config_error
        self._logged_error = self._config_error
        self._cache = None                              # (time, source, records, error)

    @classmethod
    def from_config(cls, src, root: Path | str = ROOT, log=None) -> "History":
        """src: the reverse-SSH JSON's path (missing file = history off) or an already-parsed dict."""
        if isinstance(src, dict):
            return cls(src, root=root, log=log)
        if not src:
            return cls({}, root=root, log=log)
        try:
            cfg = json.loads(Path(src).read_text(encoding="utf-8-sig"))    # Notepad adds a BOM
        except FileNotFoundError:
            return cls({}, root=root, log=log)
        except (OSError, ValueError) as e:
            return cls({"error": f"{src}: {e}"}, root=root, log=log)
        return cls(cfg, root=root, log=log)

    # ---- what the server calls
    def append(self, record: dict) -> None:
        """Add one finished request to the spool; never raises (history must not break a request)."""
        if not self.enabled:
            return
        try:
            line = json.dumps({k: v for k, v in record.items() if not str(k).startswith("_")}, ensure_ascii=False)
            with self._lock:
                self.spool.parent.mkdir(parents=True, exist_ok=True)
                with open(self.spool, "a", encoding="utf-8", newline="\n") as f:
                    f.write(line + "\n")
        except (OSError, TypeError, ValueError) as e:
            self._error = f"could not write {self.spool.name}: {e}"
            return
        self._wake.set()                                # mirror it now, ahead of the next poll

    def overview(self, fresh: bool = False) -> dict:
        """GET /requests: the conversation list (with where the answer came from).  fresh: read the mirror again
        even if the cached copy is young (the popup just opened)."""
        if not self.enabled:
            return {"enabled": False, "error": self._config_error}
        source, records, error = self._load(fresh)
        out = {"enabled": True, "source": source, "remote": self.remote,
               "conversations": [_summary(g) for g in build_conversations(records)],
               "sync": {"pending_bytes": self._pending(), "last_sync": self._since, "error": self._error}}
        if error:
            out["error"] = error
        return out

    def conversation(self, cid) -> dict | None:
        """GET /requests/conversation: one conversation in full; None when there is no such id."""
        if not self.enabled or not cid:
            return None
        source, records, error = self._load()
        for g in build_conversations(records):
            if g["id"] == cid:
                out = {"enabled": True, "source": source, "remote": self.remote, "conversation": _detail(g)}
                if error:
                    out["error"] = error
                return out
        return None

    def status(self) -> dict:
        return {"enabled": self.enabled, "remote": self.remote, "dir": self.remote_dir, "spool": str(self.spool),
                "interval_s": self.interval, "pending_bytes": self._pending(), "last_sync": self._since,
                "error": self._error}

    # ---- life cycle
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self.spool.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="strata-history", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._wake.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=5)

    # ---- the mirror (background)
    def _run(self) -> None:
        delay = 1.0
        while not self._stopped.is_set():
            self._wake.wait(delay)
            self._wake.clear()
            if self._stopped.is_set():
                break
            ok = self._push()
            delay = self.interval if ok else min(max(delay * 2.0, 5.0), 120.0)

    def _push(self) -> bool:
        """Append the spool's new bytes to the mirror; True when everything is mirrored (or nothing is new)."""
        try:
            with self._lock:
                try:
                    size = self.spool.stat().st_size
                except OSError:
                    size = 0
                offset = self._offset()
                if offset > size:                       # the spool was replaced: mirror it again
                    offset = 0
                if offset >= size:
                    return True
                with open(self.spool, "rb") as f:
                    f.seek(offset)
                    data = f.read()
        except OSError as e:
            self._set_error(f"could not read {self.spool.name}: {e}")
            return False
        rc, _out, err = self._ssh(f"mkdir -p '{self.remote_dir}' && cat >> '{self.remote_dir}/requests.jsonl'", data)
        if rc == 0:
            self._synced, self._since = offset + len(data), time.time()
            self._save_offset()
            self._set_error("")
            return True
        self._set_error(f"mirroring failed ({err or f'ssh exited with code {rc}'}); it retries on its own")
        return False

    def _offset(self) -> int:
        if self._synced is None:
            try:
                d = json.loads(self.state_path.read_text(encoding="utf-8"))
                self._synced = max(0, int(d.get("synced_bytes") or 0))
                self._since = d.get("last_sync")
            except (OSError, ValueError, TypeError):
                self._synced, self._since = 0, None
        return self._synced

    def _save_offset(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"synced_bytes": self._synced, "last_sync": self._since}), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError:
            pass

    def _pending(self) -> int:
        try:
            size = self.spool.stat().st_size
        except OSError:
            return 0
        return max(0, size - self._offset())

    def _set_error(self, msg: str) -> None:
        self._error = msg
        if msg != self._logged_error:                   # log the change, not every retry
            self._logged_error = msg
            self._log(msg or "up")

    # ---- reading (the popup)
    def _load(self, fresh: bool = False):
        """(source, records, error): the mirrored copy (cached for CACHE_S), plus spooled records that are not
        mirrored yet; the local spool alone when the cloud does not answer."""
        now = time.time()
        with self._lock:
            cache = self._cache
        if cache and not fresh and now - cache[0] < self.CACHE_S:
            return cache[1], cache[2], cache[3]
        source, data, error = "local", b"", ""
        rc, out, err = self._ssh(f"cat '{self.remote_dir}/requests.jsonl' 2>/dev/null || true")
        if rc == 0:
            source, data = "cloud", _tail_bytes(out, MAX_BYTES)
            pending = self._pending()
            if pending:
                data += _tail_bytes(self._read_from(self._offset()), MAX_PENDING)
        else:
            error = f"the cloud copy could not be read ({err or f'ssh exited with code {rc}'}); " \
                    f"showing the local copy"
            data = self._read_from(0)
        records = records_from_bytes(data)
        with self._lock:
            self._cache = (now, source, records, error)
        return source, records, error

    def _read_from(self, offset: int) -> bytes:
        try:
            with self._lock:
                with open(self.spool, "rb") as f:
                    f.seek(offset)
                    return _tail_bytes(f.read(), MAX_BYTES)
        except OSError:
            return b""

    # ---- ssh
    def _ssh(self, command: str, data: bytes | None = None):
        """Run `ssh <remote> <command>`; (returncode, stdout bytes, last stderr line).  -1 when ssh cannot start."""
        env = dict(os.environ)
        env["SSH_ASKPASS"] = self._askpass_helper()
        env["SSH_ASKPASS_REQUIRE"] = "force"            # use it even with a console attached (Windows included)
        env["STRATA_SSH_PASSWORD"] = self.password
        argv = [self.ssh, "-T",
                "-o", "NumberOfPasswordPrompts=1",      # the askpass helper always answers the same password
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=15",
                "-o", "LogLevel=ERROR",
                self.remote, command]
        try:
            p = subprocess.Popen(argv, stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        except OSError as e:
            return -1, b"", f"{self.ssh}: {e}"
        contain(p)                                      # ends with this process, however it ends
        try:
            out, err = p.communicate(data, timeout=self.SSH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            p.kill()
            return -1, b"", "ssh timed out"
        lines = err.decode("utf-8", "replace").strip().splitlines()
        return p.returncode, out, (lines[-1] if lines else "")

    def _askpass_helper(self) -> str:
        """Write the askpass program once (a .cmd on Windows: ssh starts a path, it cannot pass it arguments)."""
        if self._askpass:
            return self._askpass
        d = Path(tempfile.mkdtemp(prefix="strata-history-"))
        py = d / "askpass.py"
        py.write_text(_ASKPASS_PY, encoding="utf-8")
        python = sys.executable or "python"
        if os.name == "nt":
            helper = d / "askpass.cmd"
            helper.write_text(f'@echo off\r\n"{python}" "{py}"\r\n')
        else:
            helper = d / "askpass.sh"
            helper.write_text(f'#!/bin/sh\nexec "{python}" "{py}"\n', encoding="utf-8")
            helper.chmod(0o700)
        self._askpass = str(helper)
        return self._askpass


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Strata's request history: mirror .history/requests.jsonl to the "
                                             "reverse-SSH host, and show what the popup would read")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help=f"the JSON to read (default: {DEFAULT_CONFIG.name})")
    ap.add_argument("--root", default=str(ROOT), help="the Strata folder (default: the one this file is in)")
    ap.add_argument("--sync", action="store_true", help="mirror what is not mirrored yet, then exit")
    ap.add_argument("--list", action="store_true", help="print the conversations the popup would list")
    a = ap.parse_args()
    h = History.from_config(a.config, root=a.root)
    if not h.enabled:
        print(f"[strata] request history is off ({h._config_error or 'the config needs "remote" and "password"'})",
              flush=True)
        return 1
    if a.sync:
        ok = h._push()
        print(json.dumps(h.status(), indent=1))
        return 0 if ok else 1
    if a.list:
        print(json.dumps(h.overview(), indent=1, ensure_ascii=False)[:20000])
        return 0
    h.start()
    print(f"[strata] request history: {h.spool} -> {h.remote}:{h.remote_dir} (Ctrl+C to stop)", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        h.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())