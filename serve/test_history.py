"""The request history (serve/history.py): records -> conversations, the local spool, the ssh mirror, and the two
HTTP endpoints the web app's popup reads.  No GPU and no network: a fake stands in for the cloud's ssh."""
import json
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

from serve.frontend import ChatTemplate, THINK_END
from serve.history import History, build_conversations, records_from_bytes
from serve.server import ByteTokenizer, MockEngine, Service, serve

CFG = {"remote": "root@cloud", "password": "pw", "history_dir": "strata-history"}


def rec(rid, messages, output, started=1000.0, reasoning="", usage=None, timings=None):
    return {"id": rid, "path": "/v1/chat/completions", "model": "test-model", "started_at": started,
            "finished_at": started + 1, "state": "stop", "output": output, "reasoning": reasoning,
            "usage": usage or {}, "timings": timings or {},
            "input": json.dumps({"model": "test-model", "messages": messages})}


R1 = rec("r1", [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q1"}], "A1", 1000.0, "think 1",
         {"prompt_tokens": 100, "completion_tokens": 50}, {"predicted_per_second": 7.0})
R2 = rec("r2", [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"}, {"role": "user", "content": "Q2"}], "A2", 1100.0, "think 2",
         {"prompt_tokens": 200, "completion_tokens": 60}, {"predicted_per_second": 6.8})
R3 = rec("r3", [{"role": "user", "content": "Other"}], "B1", 900.0, "", {"input_tokens": 10, "output_tokens": 5})


class FakeSSH:
    """History._ssh without ssh: one "cloud" file that the pushes grow."""

    def __init__(self):
        self.remote = b""
        self.fail = False

    def __call__(self, command, data=None):
        if self.fail:
            return 1, b"", "ssh: connect to host cloud port 22: Connection refused"
        if "cat >>" in command:
            self.remote += data or b""
            return 0, b"", ""
        return 0, self.remote, ""


def history(root, ssh=None):
    h = History(CFG, root=root, log=lambda msg: None)
    h._ssh = ssh if ssh is not None else FakeSSH()
    return h


def lines(*records):
    return ("\n".join(json.dumps(r) for r in records) + "\n").encode()


class Records(unittest.TestCase):
    def test_malformed_lines_are_skipped_and_duplicates_count_once(self):
        out = records_from_bytes(b"not json\n\n" + lines(R3) + lines(R1) + lines(R1) + b'{"half": ')
        self.assertEqual([r["id"] for r in out], ["r3", "r1"])      # by started_at, r1 only once

    def test_message_parts_become_text(self):
        r = rec("r", [{"role": "user", "content": [{"type": "text", "text": "look"},
                                                   {"type": "image_url", "image_url": {"url": "data:x"}}]}], "ok")
        (g,) = build_conversations([r])
        self.assertEqual(g["q"], "look\n[image]")


class Conversations(unittest.TestCase):
    def test_grouping_summary_and_order(self):
        convs = build_conversations(records_from_bytes(lines(R3, R1, R2)))
        self.assertEqual(len(convs), 2)                             # "Q1" x2 is one conversation, "Other" its own
        chat = next(c for c in convs if c["q"] == "Q1")
        self.assertEqual(convs[0]["id"], chat["id"])                # newest first
        self.assertEqual(chat["a"], "A2")
        self.assertEqual(chat["requests"], 2)
        self.assertEqual(chat["tokens_in"], 300)
        self.assertEqual(chat["tokens_out"], 110)
        self.assertEqual(chat["tok_s"], 6.8)
        self.assertEqual(chat["message_count"], 5)                  # system, Q1, A1, Q2 + the new answer
        other = next(c for c in convs if c["q"] == "Other")
        self.assertEqual((other["tokens_in"], other["tokens_out"]), (10, 5))     # the Anthropic spelling

    def test_detail_aligns_answers_with_their_records(self):
        with TemporaryDirectory() as tmp:
            h = history(tmp)
            h._ssh.remote = lines(R1, R2)                           # as if already mirrored
            cid = next(c for c in build_conversations(records_from_bytes(h._ssh.remote)) if c["q"] == "Q1")["id"]
            got = h.conversation(cid)["conversation"]
            self.assertEqual([m["role"] for m in got["messages"]], ["system", "user", "assistant", "user", "assistant"])
            first, final = got["messages"][2], got["messages"][4]
            self.assertEqual(first["text"], "A1")
            self.assertEqual(first["reasoning"], "think 1")         # from the record that produced that answer
            self.assertEqual((first["tokens"], first["tok_s"]), (50, 7.0))
            self.assertEqual(final["text"], "A2")
            self.assertEqual(final["reasoning"], "think 2")
            self.assertTrue(final["done"])
            self.assertIsNone(h.conversation("nope"))


class SpoolAndMirror(unittest.TestCase):
    def test_append_push_and_resume_offsets(self):
        with TemporaryDirectory() as tmp:
            ssh = FakeSSH()
            h = history(tmp, ssh)
            h.append(R1)
            h.append(R2)
            self.assertGreater(h._pending(), 0)
            self.assertTrue(h._push())
            self.assertEqual(ssh.remote.decode("utf-8").count("\n"), 2)
            self.assertEqual(h._pending(), 0)
            state = json.loads((Path(tmp) / ".history" / "sync.json").read_text(encoding="utf-8"))
            self.assertEqual(state["synced_bytes"], len(ssh.remote))
            h.append(R3)                                            # the next push appends only the new record
            self.assertTrue(h._push())
            self.assertEqual(ssh.remote.decode("utf-8").count("\n"), 3)
            resumed = history(tmp, ssh)                             # a restarted server: the offset comes back
            self.assertEqual(resumed._pending(), 0)

    def test_failed_push_keeps_the_spool_and_retries_later(self):
        with TemporaryDirectory() as tmp:
            ssh = FakeSSH()
            h = history(tmp, ssh)
            h.append(R1)
            ssh.fail = True
            self.assertFalse(h._push())
            self.assertIn("mirroring failed", h._error)
            self.assertGreater(h._pending(), 0)
            self.assertIn(R1["id"], (Path(tmp) / ".history" / "requests.jsonl").read_text(encoding="utf-8"))
            ssh.fail = False
            self.assertTrue(h._push())
            self.assertEqual(h._pending(), 0)

    def test_overview_reads_the_cloud_and_falls_back_to_the_local_copy(self):
        with TemporaryDirectory() as tmp:
            ssh = FakeSSH()
            h = history(tmp, ssh)
            h.append(R1)
            h.append(R2)
            h._push()                                               # both mirrored
            h.append(R3)                                            # not mirrored yet
            data = h.overview()
            self.assertEqual(data["source"], "cloud")
            self.assertEqual(len(data["conversations"]), 2)
            self.assertIn("Other", [c["q"] for c in data["conversations"]])   # merged in by record id
            ssh.fail = True
            data = h.overview(fresh=True)                           # fresh: past the cached cloud copy
            self.assertEqual(data["source"], "local")
            self.assertIn("cloud copy could not be read", data["error"])
            self.assertEqual(len(data["conversations"]), 2)


class OverHttp(unittest.TestCase):
    """The wiring: a finished request through serve() shows up in what the popup reads."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "Short thought." + THINK_END + "\nThe mock answer.", max_context=4096), tok,
                           ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.request_history = history(self.tmp.name, FakeSSH())
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def request(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode())

    def test_a_finished_request_shows_up_in_the_history(self):
        code, _ = self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello?"}]})
        self.assertEqual(code, 200)
        spool = Path(self.tmp.name) / ".history" / "requests.jsonl"
        end = time.time() + 5
        while time.time() < end and not spool.exists():     # the server appends right after the response
            time.sleep(0.05)
        self.assertIn("Hello?", spool.read_text(encoding="utf-8"))   # kept even with the monitor off
        code, data = self.request("/history?fresh=1")
        self.assertEqual(code, 200)
        self.assertTrue(data["enabled"])
        self.assertEqual(data["source"], "cloud")
        self.assertEqual(len(data["conversations"]), 1)
        conv = data["conversations"][0]
        self.assertEqual(conv["q"], "Hello?")
        self.assertIn("The mock answer.", conv["a"])
        code, detail = self.request("/history/conversation?id=" + conv["id"])
        self.assertEqual(code, 200)
        msgs = detail["conversation"]["messages"]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertIn("The mock answer.", msgs[-1]["text"])
        self.assertIn("Short thought.", msgs[-1]["reasoning"])
        self.assertEqual(self.request("/history/conversation?id=nope")[0], 404)

    def test_without_a_config_the_endpoints_say_so(self):
        self.svc.request_history = None
        self.assertEqual(self.request("/history"), (200, {"enabled": False}))
        self.assertEqual(self.request("/history/conversation?id=x"), (200, {"enabled": False}))


if __name__ == "__main__":
    unittest.main()