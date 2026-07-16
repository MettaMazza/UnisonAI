"""END-TO-END test of the ACTUAL live runtime path — NOTHING STUBBED.

Drives the REAL `SFTDiscordClient.on_message` — the exact code that runs on Discord —
against EVERY real subsystem: the real Ollama teacher (LocalTeacher / gemma-4-31b), the
real word_engine, real orbit memory, real SessionManager, real correction/teaching path,
real graduation ledger. The ONLY thing that is not a live Discord socket is the input
injection itself (a FakeMessage = "the user typed this") and a FakeChannel that captures
what the bot sends back — that is the test DRIVER, not a stubbed subsystem.

Messages are driven STRICTLY SEQUENTIALLY: each on_message is fully awaited (including the
slow real teacher round-trip) before the next message is sent — exactly how a human tests
(you only reply after the teacher has answered). No concurrency is simulated because none
occurs in that usage.

Reproduces / guards the runtime failures observed 2026-07-15:
  A. session.turns must ACCUMULATE across turns (was resetting → empty history).
  B. a later turn's teacher prompt must CONTAIN a name given on an earlier turn
     (cross-turn context actually reaching the teacher).
Prints each turn's real generated reply so generation quality is visible.

Run:  PYTHONPATH=. python3 train_eval/e2e_live_path.py
(Needs Ollama running with gemma-4-31b:latest — the real live teacher. Slow by design.)
"""
import asyncio, sys, os, zlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import discord
import omni.discord_bot as bot

ALLOWED_CHANNEL_ID = 1523685773998555227   # must match on_message's gate


class FakeUser:
    def __init__(self, uid, name): self.id = uid; self.display_name = name
    def __eq__(self, o): return isinstance(o, FakeUser) and o.id == self.id
    def __hash__(self): return hash(self.id)


class FakeSentMessage:
    def __init__(self, content): self.content = content
    async def edit(self, **kw): self.content = kw.get("content", self.content); return self
    async def add_reaction(self, *a, **k): pass


class FakeChannel:
    """Captures what the bot sends. This is the test's output tap, not a subsystem stub."""
    def __init__(self, cid): self.id = cid; self.sent = []
    async def send(self, content=None, view=None, file=None, **kw):
        self.sent.append(content or ""); return FakeSentMessage(content or "")


class FakeMessage:
    def __init__(self, content, author, channel):
        self.content = content; self.author = author
        self.channel = channel; self.attachments = []
        self.id = abs(hash(content)) % (10 ** 9)


# ── Recording wrapper around the REAL teacher: does NOT change behaviour, only
#    captures the exact prompts/history the live LocalTeacher is handed, so the test
#    can assert cross-turn context reached it. Every call delegates to the real model.
class RecordingTeacher:
    def __init__(self, real):
        self._real = real
        self.ask_prompts = []; self.rate_histories = []
    def __getattr__(self, k): return getattr(self._real, k)   # delegate everything else
    def self_rate(self, *a, **k): return self._real.self_rate(*a, **k)
    def rate(self, user_msg, reply, curriculum="general", history=""):
        self.rate_histories.append(history)
        return self._real.rate(user_msg, reply, curriculum=curriculum, history=history)
    def ask(self, prompt):
        self.ask_prompts.append(prompt)
        return self._real.ask(prompt)


async def run():
    bot.init_subsystems()
    bot.word_engine.ensure_built(bot.omni_memory, "e2e_user")

    intents = discord.Intents.default(); intents.message_content = True
    client = bot.SFTDiscordClient(intents=intents)
    client.teacher = RecordingTeacher(client.teacher)      # wrap the REAL teacher
    self_user = FakeUser(999, "UnisonBot")
    bot.SFTDiscordClient.user = property(lambda s: self_user)

    author = FakeUser(2812840720, "Maria")
    channel = FakeChannel(ALLOWED_CHANNEL_ID)
    ukey = f"discord_{zlib.crc32(str(author.id).encode())}"

    script = [
        "Session start test",
        "Hello, how are you?",
        "My name is Maria, what is your name?",
        "It's going good thank you, how's yours?",
        "Do you remember my name?",
    ]

    print("=" * 70)
    for i, text in enumerate(script, 1):
        # SEQUENTIAL: fully await this turn (incl. the slow real teacher) before the next.
        await client.on_message(FakeMessage(text, author, channel))
        sess = bot.session_manager.active_sessions.get(ukey)
        n_turns = len(sess.turns) if sess else 0
        reply = channel.sent[-1] if channel.sent else "(no reply)"
        print(f"TURN {i}: {text!r}")
        print(f"   session.turns len = {n_turns}")
        print(f"   reply: {str(reply)[:200]!r}")
        print("-" * 70)

    t = client.teacher
    sess = bot.session_manager.active_sessions.get(ukey)
    hist_len = len(sess.history_log) if sess else 0
    name_in_ask = any("Maria" in p for p in t.ask_prompts[1:])
    name_in_rate = any("Maria" in h for h in t.rate_histories[1:])
    turn5_ask = t.ask_prompts[-1] if t.ask_prompts else ""
    name_in_turn5 = "Maria" in turn5_ask

    print("\n===== E2E ASSERTIONS (real runtime path, real teacher, nothing stubbed) =====")
    print(f"[{'PASS' if hist_len >= 8 else 'FAIL'}] A. history_log accumulated across turns: {hist_len} entries (expect >=8)")
    print(f"[{'PASS' if name_in_ask else 'FAIL'}] B1. a later correction prompt to teacher contains 'Maria'")
    print(f"[{'PASS' if name_in_rate else 'FAIL'}] B2. a later rating history to teacher contains 'Maria'")
    print(f"[{'PASS' if name_in_turn5 else 'FAIL'}] B3. turn-5 teacher prompt contains 'Maria'")
    print(f"\n   teacher.ask count={len(t.ask_prompts)}  rate count={len(t.rate_histories)}")
    print("\n--- turn-5 teacher prompt (first 500 chars) ---")
    print(turn5_ask[:500] if turn5_ask else "(no ask prompt recorded)")

    ok = hist_len >= 8 and name_in_ask and name_in_rate and name_in_turn5
    print("\nRESULT:", "ALL GREEN" if ok else "FAILING (bug reproduced on the real path)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
