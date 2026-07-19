import sys
import random
import discord
from discord import app_commands
import os
import io
import re
import zlib
import time

def clean_tts_text(text):
    """Strip thinking tags, asterisks, hashes, brackets, and emojis for TTS."""
    if '\x05' in text:
        text = text.split('\x05', 1)[-1]
    text = text.replace('\x04', '').replace('\x05', '')
    # Strip markdown and bracketed text
    text = re.sub(r'[*#\[\]()]', '', text)
    # Strip common text emojis
    text = re.sub(r'[:;=]-?[)(DpP\/\\|]', '', text)
    return text.strip()
import asyncio
from omni.core import fwht_2d, ifwht_2d
from omni.memory import SynapticGraph, predict_next, omni_ledger
from omni.teacher_scaffold import LocalTeacher, GraduationLedger
from omni.logging_config import discord_logger as logger, log_learning_event, chat_logger
from omni.diagnostics import GenerationDiagnostics, MemoryDiagnostics, Timer
from omni.session import SessionManager
from omni.segmentation import segment_utterance
from omni.word_engine import word_engine, generate_multiscale, tokenize, _content_words
from omni.pair_retrieval import pair_retrieval
from omni.native_transformer import native_transformer
from omni.modalities import ImageEncoder, AudioEncoder, extract_modality_segments
from omni.identity import UnisonIdentity, UserFingerprint
from omni.observer import ObserverTeacher
from omni.distill import ModelPool, DistillationEngine, CURRICULUM_GROW_EVERY
from omni.flux_gen import FluxGenerator
from omni.voice import KokoroSpeaker, WhisperListener
from omni.tools import ToolOrchestrator

# ── Global subsystems ─────────────────────────────────────────────────────
omni_memory = None
session_manager = None
image_encoder = None
audio_encoder = None
unison_identity = None
user_fingerprints = None
observer = None
model_pool = None
flux_gen = None
kokoro = None
whisper = None
engine = None
tools_orchestrator = None

class FeedbackView(discord.ui.View):
    def __init__(self, ukey, context_chars, generated_chars, context_before_prompt,
                 user_prompt_text="", response_text="", native_feedback=None,
                 rag_feedback=False):
        super().__init__(timeout=None)
        self.ukey = ukey
        self.context_chars = context_chars
        self.generated_chars = generated_chars
        self.context_before_prompt = context_before_prompt
        self.user_prompt_text = user_prompt_text
        self.response_text = response_text  # For TTS
        self.native_feedback = list(native_feedback or [])
        self.rag_feedback = bool(rag_feedback)
        
    @discord.ui.button(label="👍", style=discord.ButtonStyle.success)
    async def thumbs_up(self, interaction: discord.Interaction, button: discord.ui.Button):
        # LEARN from explicit human approval: reinforce the couplings among the reply's
        # content words (coherence_value.ep) and close the orbit (fold_orbit) — the
        # develop-over-time loop, driven by real feedback.
        try:
            for prompt, surface in self.native_feedback:
                await asyncio.to_thread(
                    native_transformer.mark_feedback, prompt, surface, True)
            if self.rag_feedback:
                await asyncio.to_thread(pair_retrieval.mark_feedback, True)
            omni_memory.fold_orbit(list(self.context_chars), ukey=self.ukey)
        except Exception:
            logger.error("thumbs_up learning failed", exc_info=True)
        await interaction.response.send_message(
            "Feedback recorded for the surface that produced this response.",
            ephemeral=True)
    
    @discord.ui.button(label="🔊", style=discord.ButtonStyle.primary)
    async def tts_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Synthesise Unison's response as speech via Kokoro TTS, or babble if missing."""
        await interaction.response.defer(ephemeral=False)
        speak_text = clean_tts_text(self.response_text)
        
        if not speak_text:
            await interaction.followup.send("Nothing to speak.", ephemeral=True)
            return

        if not kokoro.available:
            # Unison attempts to babble the audio itself
            await interaction.followup.send(f"*(Kokoro offline. Unison is attempting to babble raw audio streams for: `{speak_text[:30]}...`)*")
            
            # Record Unison's failure in Active Ledger for future Kokoro tutoring
            omni_ledger.add_prompt(self.ukey, f"[AUDIO_GENERATE] {speak_text}")
            await interaction.channel.send(f"**[Audio Babble]** Unison generated 0 valid audio tokens. Rating: BAD. Added to Active Ledger for Teacher.")
            return
        
        success, wav_bytes, duration_ms = await asyncio.get_event_loop().run_in_executor(
            None, kokoro.speak, speak_text
        )
        
        if success:
            wav_file = discord.File(io.BytesIO(wav_bytes), filename="unison_voice.wav")
            await interaction.followup.send(
                f"🔊 *Unison speaks ({duration_ms:.0f}ms audio):*",
                file=wav_file
            )
        else:
            await interaction.followup.send(f"TTS failed: {wav_bytes}", ephemeral=True)
        
    @discord.ui.button(label="👎", style=discord.ButtonStyle.danger)
    async def thumbs_down(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Topological Severing
        full_seq = self.context_chars
        omni_memory.prune_orbit(full_seq, ukey=self.ukey)
        # LEARN from disapproval: weaken the couplings among the reply's content words.
        try:
            for prompt, surface in self.native_feedback:
                await asyncio.to_thread(
                    native_transformer.mark_feedback, prompt, surface, False)
            if self.rag_feedback:
                await asyncio.to_thread(pair_retrieval.mark_feedback, False)
        except Exception:
            logger.error("thumbs_down learning failed", exc_info=True)
        
        session = session_manager.get_or_create(self.ukey)
        
        # Add CORRECTION command to Active Ledger with recent history
        correction = build_correction_prompt(session, self.user_prompt_text, self.response_text, is_thumbs_down=True)
        omni_ledger.add_prompt(self.ukey, correction, original_prompt=self.user_prompt_text)
        
        # Rollback the session's working context to BEFORE the prompt
        session.working_context = list(self.context_before_prompt)
        if len(session.turns) >= 2:
            session.turns = session.turns[:-2]
            session.turn_count = len(session.turns)
            
        await interaction.response.send_message("Trajectory severed. Context rewound. Prompt recorded in Active Ledger for tutoring.", ephemeral=True)


# The word tier is trusted for a segment only when it matched at least this many
# words of context somewhere — enough grip that it is recalling/recombining
# learned language rather than guessing from the unigram distribution. Below it,
# the segment defers to the char engine.
WORD_MIN_LEVEL = 2


def format_unison_output(out_text):
    """Format Unison's output as:  [THINKING: …]  (newline)  UNISON: …

    The \x04..\x05 reasoning trace is shown in full (it is intentional). When
    there is no thinking, only the response line is shown.
    """
    if '\x04' in out_text and '\x05' in out_text:
        _, rest = out_text.split('\x04', 1)
        thinking, answer = rest.split('\x05', 1)
        return f"**[THINKING:** {thinking.strip()} **]\n\n\nUNISON:** {answer.strip()}"
    clean = out_text.replace('\x04', ' ').replace('\x05', ' ').strip()
    return f"**UNISON:** {clean}"


def looks_repetitive(text):
    """
    True if the output has collapsed into a short repeating cycle (the 'Maria, How
    are you? Maria, How are you? …' failure). A clean answer never trips this; a
    loop does. Used to (a) stop generation early and (b) treat the loop as
    confused — its huge suffix depth otherwise fools self_rate into 'good'.
    """
    t = text.replace('\x04', '').replace('\x05', '')
    n = len(t)
    if n < 120:
        return False
    for period in range(4, 81):                 # cycle length 4..80 chars
        if 2 * period <= n and t[-period:] == t[-2 * period:-period]:
            return True
    return False


def build_correction_prompt(session, final_content, clean_eval_text, reason=None, is_thumbs_down=False):
    history = ""
    # Full conversational context so the response is coherent with the discussion
    # (e.g. it must USE a name the user already gave, not ask for it again).
    # SOURCE = the append-only history_log, NOT session.turns: turns is trimmed by the
    # correction/severing/feedback paths and was observed resetting to empty every turn
    # in the live process, so the teacher never got any context. history_log is written
    # once per completed turn and never trimmed, so it always carries the real conversation.
    prev_turns = list(session.history_log)
    logger.info(f"[correction-context] history_log={len(session.history_log)} entries, "
                f"turns={len(session.turns)} entries -> using history_log")
    if prev_turns:
        history = "Conversation so far:\n"
        for role, text in prev_turns:
            name = "User" if role == "user" else "Unison"
            history += f"{name}: {text}\n"
        history += "\n"

    # Ask the teacher for a NATURAL, in-persona response to the user's message.
    # Deliberately NOT framed as "you glitched, fix this" — that framing makes the
    # teacher reason about error-correction ("Draft 1/2/3", "Evaluator feedback",
    # "via system correction"), and that meta is what gets baked as Unison's voice.
    # A clean prompt yields clean reasoning + a clean answer.
    return (f"[CORRECTION] {history}The user says: '{final_content}'. "
            f"Respond as Unison — natural, warm, in your own voice, using the conversation above.")

async def send_chunked(channel, text, view=None):
    """Safely chunks messages to avoid Discord's 4000 character limit without breaking markdown."""
    chunk_size = 1900
    if len(text) <= chunk_size:
        if view:
            return await channel.send(text, view=view)
        else:
            return await channel.send(text)

    last_msg = None
    lines = text.split('\n')
    current_chunk = ""
    
    for line in lines:
        # If a single line is absurdly long, we have to hard split it
        if len(line) > chunk_size:
            if current_chunk:
                last_msg = await channel.send(current_chunk)
                current_chunk = ""
            for i in range(0, len(line), chunk_size):
                last_msg = await channel.send(line[i:i+chunk_size])
            continue
            
        if len(current_chunk) + len(line) + 1 > chunk_size:
            last_msg = await channel.send(current_chunk)
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
            
    if current_chunk.strip():
        if view:
            last_msg = await channel.send(current_chunk, view=view)
        else:
            last_msg = await channel.send(current_chunk)
    elif view and last_msg:
        await last_msg.edit(view=view)
            
    return last_msg

class SFTDiscordClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_task = None        # continuous self-play loop
        self.distill_task = None     # multi-model distillation loop
        self.distill_engine = None   # DistillationEngine instance
        self.diagnostic_mode = True  # ON by default
        self.voice_mode = False      # auto TTS on responses
        self.word_level = True       # word tier (fold-mix) before char fallback
        self.teacher = LocalTeacher(model_name="gemma-4-31b:latest", creativity="high")
        self.tree = app_commands.CommandTree(self)
        self.unison_last_babble = ""
        self.unison_stagnation = 0
        self.distill_running_context = []
        self.distill_history_turns = []
        self.last_interaction_time = time.time()
        self.idle_watchdog_task = None
        self.primary_channel = None
        self.primary_ukey = None
        # /auto toggles idle self-training. When enabled, the idle watchdog starts
        # tutoring/self-play once idle; when disabled, the engine just waits idle.
        self.idle_auto_enabled = False
        # The competitive graduation ladder, live (paper Sec 8.4). Every turn is a
        # head-to-head: the engine's coherence-value self-rating vs the teacher. A
        # territory graduates at p >= 1/2 (the fold lock) -> the engine becomes SOVEREIGN
        # there (stops asking the teacher, judges itself by the coherence critic) -> and
        # when broadly graduated it is its own teacher / observer.
        self.grad_ledger = GraduationLedger()
        self._graduated_announced = set()
        # Foundational data = the conversational language stores (fluency + couplings),
        # which survive /clear. Orbits (lessons + conversation) are runtime and wiped.
        # Empirical growth tracking: recent fold-coherence scores + the benchmark loop.
        from collections import deque
        self.recent_coherence = deque(maxlen=50)
        self.turn_counter = 0
        self.benchmark_task = None
        self.scrape_task = None       # /scrape autonomous conversational-data scraper
        self._register_commands()

    def _register_commands(self):
        """Register all slash commands on the command tree."""

        @self.tree.command(name="auto", description="Toggle idle self-training (tutoring/self-play when idle)")
        async def auto_cmd(interaction: discord.Interaction):
            ukey = f"discord_{zlib.crc32(str(interaction.user.id).encode())}"
            session_manager.get_or_create(ukey)
            # Toggling /auto is a user interaction — reset the idle timer so a fresh
            # ~2-minute window opens before the watchdog first triggers.
            self.last_interaction_time = time.time()
            # Remember where the idle watchdog should run.
            self.primary_channel = interaction.channel
            self.primary_ukey = ukey
            if self.idle_auto_enabled:
                # Turn idle self-training OFF, and stop any run in progress.
                self.idle_auto_enabled = False
                if self.auto_task and not self.auto_task.done():
                    self.auto_task.cancel()
                    self.auto_task = None
                    model_pool.unload_all()
                await interaction.response.send_message(
                    "**[AUTO MODE OFF]** Idle self-training disabled — Unison waits idle until you speak."
                )
            else:
                # Turn idle self-training ON. Training does not start now; the idle
                # watchdog starts it once the channel has been idle ~2 minutes, pauses
                # it whenever you speak, and resumes it when idle again.
                self.idle_auto_enabled = True
                model_pool.discover_all()
                self.distill_engine = DistillationEngine(model_pool, teacher=self.teacher)
                await interaction.response.send_message(
                    f"**[AUTO MODE ON]** Idle self-training enabled — Unison will tutor and self-play when idle "
                    f"(~2 min), and pause the moment you speak.\n"
                    f"Discovered {len(model_pool.models)} models. Use `/models` to view the distillation queue."
                )

        @self.tree.command(name="diag", description="Dump live graph statistics and system state")
        async def diag_cmd(interaction: discord.Interaction):
            ukey = f"discord_{zlib.crc32(str(interaction.user.id).encode())}"
            total_orbits, total_chars = MemoryDiagnostics.log_graph_stats(omni_memory)
            session = session_manager.get_or_create(ukey)
            diag_msg = (
                f"```\n"
                f"── Unison Diagnostics ───────────────\n"
                f"  Graph Orbits     : {total_orbits}\n"
                f"  Graph Characters : {total_chars}\n"
                f"  User Keys        : {list(omni_memory.orbits.keys())}\n"
                f"  Session ID       : {session.session_id}\n"
                f"  Session Turns    : {session.turn_count}\n"
                f"  Working Memory   : {len(session.working_context)} chars\n"
                f"  Active Ledger Queue : {len(omni_ledger.pending_prompts)}\n"
                f"  Diagnostic Mode  : {'ON' if self.diagnostic_mode else 'OFF'}\n"
                f"  Active Sessions  : {len(session_manager.active_sessions)}\n"
                f"─────────────────────────────────────\n"
                f"```"
            )
            await interaction.response.send_message(diag_msg)

        @self.tree.command(name="new", description="End current conversation session, bank it as a coherent orbit, and start fresh")
        async def new_cmd(interaction: discord.Interaction):
            ukey = f"discord_{zlib.crc32(str(interaction.user.id).encode())}"
            
            # End and bank the current session
            full_sequence = session_manager.end_session(ukey)
            banked = False
            if full_sequence and len(full_sequence) > 10:
                omni_memory.hold_orbit(full_sequence, ukey=ukey)
                banked = True
            
            # Start a fresh session
            new_session = session_manager.start_session(ukey)
            
            turns = 0
            chars = 0
            if full_sequence:
                turns = sum(1 for c in full_sequence if c == '\x02')
                chars = len(full_sequence)
            
            await interaction.response.send_message(
                f"```\n"
                f"── New Session Started ──────────────\n"
                f"  Previous Session : {'Banked as coherent orbit ✓' if banked else 'Too short to bank'}\n"
                f"  Turns Banked     : {turns}\n"
                f"  Characters       : {chars}\n"
                f"  New Session ID   : {new_session.session_id}\n"
                f"  Working Memory   : 0 chars (fresh)\n"
                f"─────────────────────────────────────\n"
                f"```"
            )

        @self.tree.command(name="clear", description="Factory reset — wipe all memory, contexts, logs, and start fresh")
        async def clear_cmd(interaction: discord.Interaction):
            # 1. Kill auto loop if running
            if self.auto_task and not self.auto_task.done():
                self.auto_task.cancel()
                self.auto_task = None
            
            # 2. End all sessions
            session_manager.active_sessions = {}

            # 3. WIPE the orbits (taught lessons + conversation memory). These are NOT
            #    foundational — they are what accumulates at runtime. Blank slate for a
            #    fresh monitored test.
            omni_memory.orbits = {}
            omni_memory._rebuild_caches()
            if os.path.exists(omni_memory.save_path):
                os.remove(omni_memory.save_path)
            word_engine._sig = None   # force the word tier to rebuild (now lesson-free)

            # 4. Reset the runtime competitive tally + session growth counters.
            omni_ledger.clear()
            self.grad_ledger = GraduationLedger()
            self._graduated_announced = set()
            self.recent_coherence.clear()
            self.turn_counter = 0

            # The FOUNDATIONAL data — the CONVERSATIONAL LANGUAGE the engine speaks from —
            # is preserved and NEVER touched by /clear:
            #   - omni/word_fluency.pkl   (conversational fluency, from the downloaded chat data)
            #   - omni/word_coupling.pkl  (conversational couplings + accumulated learning)
            #   - the logs (kept, so the growth trajectory stays continuous)

            logger.info("/clear: orbits wiped; conversational language (fluency + couplings) preserved.")
            await interaction.response.send_message(
                "```\n"
                "══════════════════════════════════════\n"
                "  RUNTIME CLEARED — language intact\n"
                "══════════════════════════════════════\n"
                "  ✓ Orbits wiped (taught lessons + conversation)\n"
                "  ✓ Conversational fluency PRESERVED (foundation)\n"
                "  ✓ Conversational couplings PRESERVED (foundation + learning)\n"
                "  ✓ Graduation race reset (fresh)\n"
                "  ✓ Logs kept (trajectory continuous)\n"
                "\n"
                "  Blank conversation slate; full language ability. Ready to test.\n"
                "══════════════════════════════════════\n"
                "```"
            )

        @self.tree.command(name="diagnostic", description="Toggle per-message latency diagnostics display on/off")
        async def diagnostic_cmd(interaction: discord.Interaction):
            self.diagnostic_mode = not self.diagnostic_mode
            state = "ON" if self.diagnostic_mode else "OFF"
            await interaction.response.send_message(f"**[Diagnostic Mode: {state}]** Per-message latency diagnostics are now {state.lower()}.")

        @self.tree.command(name="scrape", description="Autonomously scrape high-quality conversational datasets and grow the language foundation")
        async def scrape_cmd(interaction: discord.Interaction):
            if getattr(self, "scrape_task", None) and not self.scrape_task.done():
                await interaction.response.send_message("**[Scrape]** Already running — I'll post progress as it goes.")
                return
            await interaction.response.send_message(
                "**[Scrape]** Discovering high-quality conversational datasets and growing the language "
                "foundation… runs in the background (chat stays live); I'll post progress and rebuild the "
                "fluency + coupling stores when done. Orbits are untouched — this only grows the foundation.")
            channel = interaction.channel
            loop = asyncio.get_event_loop()
            def progress(msg):
                try:
                    asyncio.run_coroutine_threadsafe(channel.send(f"*(scrape)* {str(msg)[:1800]}"), loop)
                except Exception:
                    pass
            async def run():
                try:
                    from omni import scraper
                    added, nbytes = await asyncio.to_thread(scraper.scrape_and_extend, 200_000_000, 6, progress)
                    word_engine.reload_language_stores()
                    word_engine._sig = None
                    if added:
                        await channel.send(
                            f"**[Scrape complete]** +{len(added)} datasets, +{nbytes//1_000_000}MB conversational "
                            f"data. Foundation rebuilt & hot-reloaded. New: {', '.join(added)}")
                    else:
                        await channel.send("**[Scrape complete]** No new datasets (all known high-quality sets already scraped).")
                except Exception as e:
                    logger.error("scrape failed", exc_info=True)
                    await channel.send(f"**[Scrape error]** {str(e)[:300]}")
            self.scrape_task = asyncio.create_task(run())


        @self.tree.command(name="models", description="Show available models and distillation progress")
        async def models_cmd(interaction: discord.Interaction):
            model_pool.discover_all()
            summary = model_pool.summary()
            status = ""
            if hasattr(self, 'distill_engine') and self.distill_engine:
                s = self.distill_engine.status()
                status = (
                    f"\n── Distillation Status ──\n"
                    f"  Total iterations: {s['total_iterations']}\n"
                    f"  Current model: {s['current_model']}\n"
                    f"  Models remaining: {s['models_remaining']}/{s['models_total']}\n"
                    f"  Stagnation streak: {s['stagnation_streak']}\n"
                )
            full_text = f"```\n{summary}{status}\n```"
            if len(full_text) > 1900:
                with io.BytesIO(f"{summary}{status}".encode('utf-8')) as f:
                    await interaction.response.send_message(
                        "Models summary is too long for a single message. See attached file:",
                        file=discord.File(f, filename="models_summary.txt")
                    )
            else:
                await interaction.response.send_message(full_text)

        @self.tree.command(name="voice", description="Toggle automatic TTS on Unison's responses")
        async def voice_cmd(interaction: discord.Interaction):
            self.voice_mode = not self.voice_mode
            state = "ON" if self.voice_mode else "OFF"
            avail = "✓" if kokoro.available else "✗ (Kokoro not loaded)"
            await interaction.response.send_message(
                f"**[Voice Mode: {state}]** Auto TTS {avail}"
            )

    async def on_ready(self):
        logger.info(f"SFT Omni Architecture online as {self.user}")
        # Sync slash commands with Discord instantly for all connected guilds
        try:
            synced_total = 0
            for guild in self.guilds:
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                synced_total += len(synced)
            logger.info(f"Synced {synced_total} slash commands instantly to {len(self.guilds)} guilds.")
        except Exception as e:
            logger.error(f"Failed to sync slash commands: {e}", exc_info=True)
        logger.info("Awaiting input to begin exact derivations.")
        # Foundational data = the CONVERSATIONAL LANGUAGE stores (word_fluency.pkl +
        # word_coupling.pkl), preserved across /clear. Orbits are runtime and are wiped.
        self.idle_watchdog_task = asyncio.create_task(self._idle_watchdog())
        self.benchmark_task = asyncio.create_task(self._benchmark_loop())

    async def _benchmark_loop(self):
        """Empirical scaling test: periodically measure the growth trajectory —
        learned couplings, live coherence, memory, graduation — log it and post it
        to the channel so progress is tracked over time."""
        import json as _json, datetime as _dt
        from omni.logging_config import LOG_DIR
        await self.wait_until_ready()
        traj_path = os.path.join(LOG_DIR, "benchmark_trajectory.jsonl")
        INTERVAL = 900  # 15 min
        while not self.is_closed():
            await asyncio.sleep(INTERVAL)
            try:
                g = word_engine._load_coupling()
                words = len(g)
                edges = sum(len(v) for v in g.values())
                coh = list(self.recent_coherence)
                avg_coh = sum(coh) / len(coh) if coh else 0.0
                orbits = sum(len(v) for v in omni_memory.orbits.values())
                gw, gl = self.grad_ledger.scores.get("general", [0, 0])
                gp = gw / (gw + gl) if (gw + gl) else 0.0
                sq = word_engine._load_span_quality()
                good_spans = sum(1 for v in sq.values() if v > 0)
                bad_spans = sum(1 for v in sq.values() if v < 0)
                rec = {"ts": _dt.datetime.utcnow().isoformat() + "Z", "coupling_words": words,
                       "coupling_edges": edges, "avg_coherence": round(avg_coh, 3),
                       "orbits": orbits, "grad_wins": gw, "grad_losses": gl,
                       "grad_p": round(gp, 3), "turns": self.turn_counter,
                       "good_spans": good_spans, "bad_spans": bad_spans}
                with open(traj_path, "a") as f:
                    f.write(_json.dumps(rec) + "\n")
                chat_logger.info(f"[BENCHMARK] {rec}")
                if self.primary_channel and self.turn_counter > 0:
                    grad = "SOVEREIGN ✓" if self.grad_ledger.has_graduated("general") else f"{gw}/{gw+gl} (need p≥0.5)"
                    await self.primary_channel.send(
                        f"**[Growth benchmark]** couplings {words:,} words / {edges:,} edges | "
                        f"spans learned +{good_spans}/-{bad_spans} | live coherence {avg_coh:.2f} (lock 0.50) | "
                        f"memory {orbits} orbits | graduation {grad} | {self.turn_counter} turns")
            except Exception:
                logger.error("benchmark loop error", exc_info=True)

    async def _idle_watchdog(self):
        """When idle self-training is enabled (via /auto), start the tutoring/
        self-play loop once the channel has been idle ~2 minutes. When it is
        disabled, the engine simply waits idle."""
        while True:
            await asyncio.sleep(5)
            if self.idle_auto_enabled and self.primary_channel and self.primary_ukey:
                if time.time() - self.last_interaction_time > 120:
                    if self.auto_task is None or self.auto_task.done():
                        model_pool.discover_all()
                        self.distill_engine = DistillationEngine(model_pool, teacher=self.teacher)
                        self.auto_task = asyncio.create_task(self._auto_loop(self.primary_channel, self.primary_ukey))
                        await self.primary_channel.send(
                            f"*(Idle for 2 mins)* **[AUTO MODE ON]** Spawning continuous asynchronous tutoring and distillation engine.\n"
                        )
                        
    async def _auto_loop(self, channel, ukey):
        """
        Continuous asynchronous background task for Tutoring and Self-Play.
        """
        try:
            # (a) Fresh curriculum for this session — generated in a WORKER THREAD
            # so the long generation never blocks the Discord event loop/heartbeat.
            if self.distill_engine is not None and getattr(self.distill_engine, "teacher", None) is not None:
                if self.distill_engine.curriculum_is_empty():
                    # First run only — author the initial curriculum. Regenerating on
                    # every /auto would be wasted compute; a fresh batch otherwise comes
                    # ONLY once the current curriculum has been fully worked through.
                    await channel.send("*(Background)* No curriculum yet — generating the first batch…")
                    try:
                        totals = await asyncio.to_thread(self.distill_engine.refresh_all)
                        if totals:
                            await channel.send(
                                "*(Background)* Curriculum ready — "
                                + ", ".join(f"{a}: {n}" for a, n in totals.items())
                                + " teacher-generated seeds."
                            )
                    except Exception as e:
                        logger.error("Curriculum generation failed", exc_info=True)
                else:
                    counts = self.distill_engine.curriculum_counts()
                    await channel.send(
                        "*(Background)* Resuming the existing curriculum ("
                        + ", ".join(f"{a}: {n}" for a, n in counts.items())
                        + ") — no regeneration until it's completed."
                    )

            iteration = 0
            while True:
                # Curriculum growth: author a fresh batch ONLY when the current
                # curriculum has been fully worked through (a completed pass) — never
                # on a fixed interval, so no teacher compute is wasted regenerating
                # seeds that haven't been used yet.
                if self.distill_engine and self.distill_engine.take_pass_completed():
                    grown = await asyncio.to_thread(self.distill_engine.grow_curricula)
                    if grown:
                        await channel.send(
                            f"*(Background)* Curriculum completed — authored a fresh batch; "
                            f"`{grown[0]}` now has {grown[1]} teacher-generated seeds."
                        )

                # Phase A: Targeted Tutoring
                if omni_ledger.pending_prompts:
                    await channel.send(f"*(Background)* Processing {len(omni_ledger.pending_prompts)} pending prompts with Teacher...")
                    learned_count = 0
                    for item in omni_ledger.pending_prompts:
                        if item['ukey'] == ukey:
                            prompt_text = item['context']
                            original_prompt = item.get('original_prompt', prompt_text)
                            
                            if prompt_text.startswith("[AUDIO_GENERATE]"):
                                if not kokoro.available:
                                    continue  # Cannot teach audio without Kokoro
                                
                                audio_text = prompt_text.replace("[AUDIO_GENERATE]", "").strip()
                                success, wav_bytes, _ = await asyncio.to_thread(kokoro.speak, audio_text)
                                if not success:
                                    continue
                                    
                                audio_chars = audio_encoder.encode(wav_bytes)
                                prompt_chars = ['\x02'] + list(prompt_text) + ['\x03']
                                teacher_chars = audio_chars + ['\x02']
                                omni_memory.hold_orbit(prompt_chars + teacher_chars, ukey=ukey)
                                learned_count += 1
                                
                                log_learning_event("bad_ledger_audio", prompt_text, f"<{len(audio_chars)} audio chars>", "", "", ukey=ukey)
                                
                                display_ans = f"*[Kokoro synthesized {len(audio_chars)} exact audio tokens]*"
                                full_msg = f"**[Tutoring - Active Learning (Audio)]**\n**Prompt:** {prompt_text}\n**Teacher Mapped:** {display_ans}"
                                await send_chunked(channel, full_msg)
                            
                            elif prompt_text.startswith("[IMAGE_GENERATE]"):
                                if not flux_gen.available:
                                    continue  # Cannot teach image without Flux
                                
                                img_prompt = prompt_text.replace("[IMAGE_GENERATE]", "").strip()
                                success, result_bytes = await asyncio.to_thread(flux_gen.generate, img_prompt)
                                if not success:
                                    continue
                                
                                img_chars = await asyncio.to_thread(image_encoder.encode, result_bytes)
                                prompt_chars = ['\x02'] + list(prompt_text) + ['\x03']
                                teacher_chars = img_chars + ['\x02']
                                omni_memory.hold_orbit(prompt_chars + teacher_chars, ukey=ukey)
                                learned_count += 1
                                
                                log_learning_event("bad_ledger_image", prompt_text, f"<{len(img_chars)} image chars>", "", "", ukey=ukey)
                                
                                display_ans = f"*[Flux generated {len(img_chars)} exact image tokens]*"
                                full_msg = f"**[Tutoring - Active Learning (Image)]**\n**Prompt:** {prompt_text}\n**Teacher Mapped:** {display_ans}"
                                await send_chunked(channel, full_msg)
                                
                            elif prompt_text.startswith("[TOOL_USE]"):
                                # Silent Distillation in background thread to avoid blocking heartbeat
                                instruction = (
                                    "You are an AI that uses tools. Respond ONLY with a strictly formatted JSON block "
                                    "containing 'tool' and 'args' keys for the user's request. No conversational text.\n"
                                    f"Request: {prompt_text.replace('[TOOL_USE]', '').strip()}"
                                )
                                teacher_ans = await asyncio.to_thread(self.teacher.ask, instruction)
                                
                                clean_teacher_ans = teacher_ans
                                
                                prompt_chars = ['\x02'] + list(prompt_text) + ['\x03']
                                teacher_chars = list(clean_teacher_ans) + ['\x02']
                                omni_memory.hold_orbit(prompt_chars + teacher_chars, ukey=ukey)
                                learned_count += 1
                                
                                log_learning_event("bad_ledger_tool", prompt_text, teacher_ans, "", teacher_ans, ukey=ukey)
                                
                                display_ans = teacher_ans.replace('\x04', '*💭 Thinking:*\n> *').replace('\x05', '*\n\n')
                                full_msg = f"**[Tutoring - Active Learning (Tool)]**\n**Prompt:** {prompt_text}\n**Teacher Mapped:** {display_ans}"
                                await send_chunked(channel, full_msg)
                                
                            elif prompt_text.startswith("[CORRECTION]"):
                                instruction = prompt_text.replace("[CORRECTION]", "").strip()
                                # Ask the teacher the clean prompt directly (no "System
                                # Correction:" framing) so its reasoning is natural
                                # conversation, not error-fixing meta.
                                teacher_ans = await asyncio.to_thread(self.teacher.ask, instruction)
                                
                                clean_teacher_ans = teacher_ans
                                
                                prompt_chars = ['\x02'] + list(original_prompt) + ['\x03']
                                teacher_chars = list(clean_teacher_ans) + ['\x02']
                                omni_memory.hold_orbit(prompt_chars + teacher_chars, ukey=ukey)
                                learned_count += 1
                                
                                log_learning_event("bad_ledger_correction", original_prompt, teacher_ans, "", teacher_ans, ukey=ukey)
                                
                                display_ans = teacher_ans.replace('\x04', '*💭 Thinking:*\n> *').replace('\x05', '*\n\n')
                                full_msg = f"**[Tutoring - Live Session Correction]**\n**Original User Prompt:** {original_prompt}\n**Teacher Correction:** {display_ans}"
                                await send_chunked(channel, full_msg)
                                
                            elif prompt_text.startswith("[CONFUSION FALLBACK]"):
                                instruction = prompt_text.replace("[CONFUSION FALLBACK]", "").strip()
                                # Ask the clean instruction directly (no "System
                                # Correction:" framing) so the teacher's reasoning is
                                # natural, not error-fixing meta that gets baked as voice.
                                teacher_ans = await asyncio.to_thread(self.teacher.ask, instruction)
                                
                                clean_teacher_ans = teacher_ans
                                
                                prompt_chars = ['\x02'] + list("I'm confused.") + ['\x03']
                                teacher_chars = list(clean_teacher_ans) + ['\x02']
                                omni_memory.hold_orbit(prompt_chars + teacher_chars, ukey=ukey)
                                learned_count += 1
                                
                                log_learning_event("bad_ledger_fallback", instruction, teacher_ans, "", teacher_ans, ukey=ukey)
                                
                                display_ans = teacher_ans.replace('\x04', '*💭 Thinking:*\n> *').replace('\x05', '*\n\n')
                                full_msg = f"**[Tutoring - Confusion Fallback]**\n**Instruction:** {instruction}\n**Teacher Mapped:** {display_ans}"
                                await send_chunked(channel, full_msg)
                            else:
                                # Silent Distillation in background thread to avoid blocking heartbeat
                                teacher_ans = await asyncio.to_thread(self.teacher.ask, prompt_text)
                                
                                clean_teacher_ans = teacher_ans
                                
                                prompt_chars = ['\x02'] + list(original_prompt) + ['\x03']
                                teacher_chars = list(clean_teacher_ans) + ['\x02']
                                omni_memory.hold_orbit(prompt_chars + teacher_chars, ukey=ukey)
                                learned_count += 1
                                
                                # Log learning event
                                log_learning_event("bad_ledger_tutoring", prompt_text, teacher_ans, "", teacher_ans, ukey=ukey)
                                
                                # Full Transparency Broadcast
                                display_ans = teacher_ans.replace('\x04', '*💭 Thinking:*\n> *').replace('\x05', '*\n\n')
                                full_msg = f"**[Tutoring - Active Learning]**\n**Prompt:** {prompt_text}\n**Teacher Mapped:** {display_ans}"
                                await send_chunked(channel, full_msg)
                            
                    omni_ledger.clear()
                    await channel.send(f"*(Background)* Successfully mapped {learned_count} exact corrections.")

                # Phase B: Distillation Engine Loop (Unified)
                if not hasattr(self, 'distill_engine') or not self.distill_engine:
                    # If not initialised, initialize it
                    model_pool.discover_all()
                    self.distill_engine = DistillationEngine(model_pool, teacher=self.teacher)
                    
                model = self.distill_engine.get_current_model()
                if not model:
                    await channel.send("**[AUTO MODE]** All discovered models have been distilled! Unison is complete.")
                    self.auto_task = None
                    return
                
                # Load model (spins up llama-server if GGUF)
                api_url, model_name = model_pool.load_for_inference(model)
                if not api_url:
                    await channel.send(f"**[AUTO MODE]** Failed to load model {model.model_id}. Skipping.")
                    model.distilled = True
                    self.distill_engine.advance("")
                    continue
                
                # Get next curriculum seed
                seed_prompt, curriculum_name = self.distill_engine.get_next_seed(model)
                
                # Image generation for creative curriculum
                image_chars = []
                image_path = None
                if curriculum_name == "creative" and flux_gen.available:
                    # Parse image prompt from seed (e.g. "Describe a sunset" -> "a sunset")
                    img_prompt = seed_prompt.replace("Describe ", "").replace("Tell me about ", "")
                    success, result = await asyncio.to_thread(flux_gen.generate, img_prompt)
                    if success:
                        encoded = await asyncio.to_thread(image_encoder.encode, result)
                        image_chars.extend(encoded)
                        # The seed prompt becomes "Describe this image: [prompt]"
                        seed_prompt = f"Describe this image: {img_prompt}"
                        
                        # Send image to Discord
                        images = flux_gen.list_generated()
                        if images:
                            image_path = os.path.join(flux_gen._OUTPUT_DIR, images[-1])
                
                # ── Teacher answers ──
                creativity = "high" if curriculum_name == "creative" else "medium"
                self.teacher.set_creativity(creativity)
                
                # Context-Aware Prompting
                history_text = "".join(self.distill_history_turns)
                if history_text:
                    teacher_prompt = f"Conversation History:\n{history_text}\nUser: {seed_prompt}\nUnison:"
                else:
                    teacher_prompt = seed_prompt
                
                teacher_ans = await asyncio.to_thread(
                    self.distill_engine.query_model,
                    api_url, model_name, teacher_prompt,
                    system=self.teacher.get_system_instruction(),
                    temperature=self.teacher.temperature,
                    top_p=self.teacher.top_p,
                    repeat_penalty=self.teacher.repeat_penalty
                )
                
                if not teacher_ans:
                    await asyncio.sleep(5)
                    continue
                
                clean_ans = teacher_ans
                
                # Bank the teacher's answer into the continuous context
                prompt_chars = ['\x02'] + list(seed_prompt) + ['\x03']
                teacher_chars = list(clean_ans) + ['\x02']
                
                self.distill_running_context.extend(image_chars + prompt_chars + teacher_chars)
                self.distill_history_turns.append(f"User: {seed_prompt}\nUnison: {clean_ans}\n")
                
                # Orbit the entire sliding window!
                omni_memory.hold_orbit(self.distill_running_context, ukey=ukey)
                
                # ── Unison COMPOSES (never verbatim) ──
                # Self-play generates from the FOUNDATION exactly like the live path — it
                # NEVER char-recalls from memory. Build a schema from the running context +
                # the seed, then retrieve+recombine (fallback: the word-tier unfold).
                test_context = self.distill_running_context[:-len(teacher_chars)]
                _sch_txt = ("".join(test_context).replace("\x02", " ").replace("\x03", " ")
                            .replace("\x04", " ").replace("\x05", " "))
                schema = _content_words(tokenize(_sch_txt)) + _content_words(tokenize(seed_prompt))
                _rng = random.Random()
                babble_text = await asyncio.to_thread(word_engine.retrieve_and_compose, schema, _rng)
                if not babble_text:
                    babble_text = await asyncio.to_thread(word_engine.unfold_response, schema, _rng)
                babble_chars = list(babble_text)

                # ── Parroting Detection ──
                if babble_text == self.unison_last_babble and babble_text.strip():
                    self.unison_stagnation += 1
                else:
                    self.unison_stagnation = 0
                self.unison_last_babble = babble_text

                # ── Ratings: the FOLD coherence critic (no verbatim-depth heuristic) ──
                _content = _content_words(tokenize(babble_text))
                _ws, _cs = word_engine.coherence_score(_content, _content_words(tokenize(_sch_txt)))
                avg_depth, avg_cands = _ws, _cs   # reuse the log fields for the fold coherence
                self_rating = "good" if _ws >= 0.5 else "bad"
                self_reason = f"fold coherence word_scale={_ws:.2f}"
                
                if babble_text.strip():
                    clean_babble = babble_text
                    if '\x05' in clean_babble:
                        clean_babble = clean_babble.split('\x05')[-1].strip()
                        
                    teacher_rating, teacher_reason = await asyncio.to_thread(
                        self.teacher.rate, seed_prompt, clean_babble, curriculum_name
                    )
                else:
                    teacher_rating, teacher_reason = "bad", "Empty response"
                
                if self.unison_stagnation >= 3:
                    teacher_rating = "bad"
                    teacher_reason = "Parroting loop detected."
                
                agree = (self_rating == teacher_rating)
                
                # ── Confusion Fallback & Pruning ──
                if teacher_rating == "bad" or self.unison_stagnation >= 3:
                    # A failed seed queues a correction for tutoring — but do NOT
                    # reset the curriculum pointer to 0. Resetting on every failure
                    # (an infant fails constantly) pinned self-play in the first
                    # curriculum forever and it never reached learning/questioning/sft.
                    # The seed still advances normally via advance()'s budget below.
                    fallback_prompt = f"[CONFUSION FALLBACK] The user said: '{seed_prompt}'. Respond as Unison — naturally admit if you're unsure and ask ONE curious clarifying question so you can learn."
                    omni_ledger.add_prompt(ukey, fallback_prompt)
                elif babble_text.strip():
                    omni_ledger.add_prompt(ukey, seed_prompt)
                    if teacher_rating == "good":
                        omni_memory.fold_orbit(self.distill_running_context, ukey=ukey)
                    
                # ── Context Window Management (budget from the model's own reported window) ──
                _ctx_budget = self.teacher.context_window   # provider-reported; longform, not a hardcoded 8000
                if len(self.distill_running_context) > _ctx_budget:
                    while len(self.distill_running_context) > _ctx_budget:
                        try:
                            # Truncate at the oldest turn boundary to maintain clean state
                            idx = self.distill_running_context[1:].index('\x02') + 1
                            self.distill_running_context = self.distill_running_context[idx:]
                            if self.distill_history_turns:
                                self.distill_history_turns.pop(0)
                        except ValueError:
                            self.distill_running_context = []
                            self.distill_history_turns = []
                            break
                
                # ── Observer Checks ──
                observer.record(clean_ans, avg_depth, self_rating, teacher_rating, agree)
                observer_note = None
                
                if observer.should_check():
                    findings = observer.check()
                    if findings["interventions"]:
                        observer_note = f"Interventions: {findings['interventions']}"
                        if "switch_model" in findings["interventions"]:
                            model.distilled = True
                        if "cleanup_bad_ledger" in findings["interventions"]:
                            # Trim ledger to keep quality high
                            if len(omni_ledger.pending_prompts) > 50:
                                omni_ledger.pending_prompts = omni_ledger.pending_prompts[-50:]
                                omni_ledger.save()
                
                # Log iteration
                self.distill_engine.log_iteration(
                    model.model_id, curriculum_name, seed_prompt, clean_ans,
                    self_rating, teacher_rating, observer_note
                )
                
                # Full Transparency Broadcast
                agree_emoji = "✅" if agree else "⚠️"
                display_teacher = clean_ans[:300] + ('...' if len(clean_ans) > 300 else '')
                display_babble = babble_text[:200] + ('...' if len(babble_text) > 200 else '' if babble_text else '*(empty)*')
                
                full_msg = (
                    f"**[Auto #{self.distill_engine.total_iterations}]** Model: `{model.model_id}` | Curr: `{curriculum_name}`\n"
                    f"**Prompt:** *{seed_prompt}*\n"
                    f"**Teacher:** {display_teacher}\n"
                    f"**Unison:** {display_babble}\n"
                    f"**Ratings:** {agree_emoji} Self={self_rating} ({self_reason}) | Teacher={teacher_rating} ({teacher_reason})"
                )
                
                if image_path and os.path.exists(image_path):
                    file = discord.File(image_path, filename="flux_gen.png")
                    await channel.send(full_msg, file=file)
                else:
                    await send_chunked(channel, full_msg)
                
                # Advance engine state
                action = self.distill_engine.advance(clean_ans)
                if action == "next_model" or action == "next_curriculum":
                    await channel.send(f"*(Observer)* Engine advanced: {action}")
                

                # ── Phase C: Exact-Arithmetic Generative Topology (Self-Play) ──
                if iteration % 2 == 0:
                    self_play_context = ['\x02'] + list(seed_prompt) + ['\x03']
                    self_play_turns = 0
                    max_self_play_turns = 8
                    
                    sp_history = ""
                    loop_detected = False
                    
                    while self_play_turns < max_self_play_turns:
                        def _sp_schema(ctx):
                            t = ("".join(ctx).replace("\x02", " ").replace("\x03", " ")
                                 .replace("\x04", " ").replace("\x05", " "))
                            return _content_words(tokenize(t))

                        # Unison-A COMPOSES from the foundation (never verbatim recall)
                        _scha = _sp_schema(self_play_context)
                        text_a = await asyncio.to_thread(word_engine.retrieve_and_compose, _scha, random.Random())
                        if not text_a:
                            text_a = await asyncio.to_thread(word_engine.unfold_response, _scha, random.Random())
                        text_a = (text_a or "").strip()
                        if not text_a:
                            break

                        sp_history += f"**A:** {text_a}\n"
                        self_play_context.extend(list(text_a) + ['\x02'])

                        # Unison-B COMPOSES from the foundation (never verbatim recall)
                        _schb = _sp_schema(self_play_context)
                        text_b = await asyncio.to_thread(word_engine.retrieve_and_compose, _schb, random.Random())
                        if not text_b:
                            text_b = await asyncio.to_thread(word_engine.unfold_response, _schb, random.Random())
                        text_b = (text_b or "").strip()
                        if not text_b:
                            break
                            
                        sp_history += f"**B:** {text_b}\n"
                        self_play_context.extend(list(text_b) + ['\x02'])
                        self_play_turns += 1
                        
                        # Exact loop detection
                        if text_a == text_b or (self_play_turns > 2 and text_a in text_b):
                            loop_detected = True
                            break
                            
                    # Referee Action
                    if loop_detected:
                        omni_memory.prune_orbit(self_play_context, ukey)
                        if self.diagnostic_mode:
                            await channel.send(f"**[Generative Topology]** Loop detected and Pruned after {self_play_turns} turns.")
                    elif self_play_turns >= 4:
                        omni_memory.fold_orbit(self_play_context, ukey)
                        await send_chunked(channel, f"**[Generative Topology]** Sustained {self_play_turns} novel turns. Orbit Folded (Thickened).\n{sp_history}")
                
                iteration += 1
                    
                # Yield to the asyncio event loop to keep Discord heartbeat alive
                await asyncio.sleep(2)
                
        except asyncio.CancelledError:
            await channel.send("**[AUTO MODE OFF]** Background continuous tutoring and self-play terminated.")
        except Exception as e:
            logger.error("Error in auto loop", exc_info=True)
            await channel.send(f"**[Auto Loop Error]** {str(e)}")

    async def _generate_fragment(self, start_context, session, ukey, diag):
        """Generate one response fragment for a single segment's context.

        DEAD / NEUTRALISED: this per-character fold walk was longest-suffix memory
        recall = verbatim replay. Generation is now foundation-only (retrieve_and_compose
        / unfold_response) and this is never called. It returns nothing so that NO
        verbatim path can ever execute — there is no place verbatim recall can be used.
        """
        return ""
        generated_chars = []
        current_context = list(start_context)
        last_k = None
        while len(generated_chars) < 4000:  # safety bound
            t0 = time.perf_counter()
            max_k = last_k + 1 if last_k is not None else None

            # Query BOTH episodic (session) memory and the global graph; use
            # whichever matches DEEPER. Episodic wins only strictly deeper — on a
            # tie the global graph (taught corrections) wins over the session's
            # own echo. (See the anchor/tie-break history in on_message.)
            epi_char, epi_len, epi_cands = await asyncio.to_thread(
                predict_next, current_context, session.episodic_memory, ukey, max_k
            )
            omni_char, omni_len, omni_cands = await asyncio.to_thread(
                predict_next, current_context, omni_memory, ukey, max_k
            )
            if epi_char is not None and epi_len > 0 and epi_len > omni_len:
                next_char, suffix_len, num_cands = epi_char, epi_len, epi_cands
            else:
                next_char, suffix_len, num_cands = omni_char, omni_len, omni_cands
            char_ms = (time.perf_counter() - t0) * 1000.0

            if next_char is None:
                break

            diag.record_char(char_ms, suffix_len, num_cands)
            generated_chars.append(next_char)
            current_context.append(next_char)
            last_k = suffix_len

            # Stop if it predicts the start of a user turn.
            if generated_chars[-1] == '\x02':
                generated_chars.pop()
                break

            # Stop early if the fragment collapsed into a repeating cycle.
            if len(generated_chars) >= 120 and len(generated_chars) % 40 == 0 \
                    and looks_repetitive("".join(generated_chars)):
                break

        return "".join(generated_chars)

    async def _generate_fragment_multiscale(self, seg_context, session, ukey, diag,
                                            rng=None, trace=None):
        """Run the native causal-transformer conversational stage.

        The historical word/generic fallback is retired. It was an agent-authored
        interpretation, not a derived Unison component. The native generalisation
        route is the one-to-one causal-transformer translation. No retrieval,
        template, or generic fallback is admitted after native generation.
        """
        seg_text = ("".join(str(c) for c in seg_context)
                    .replace("\x02", " ").replace("\x03", " ").strip())
        try:
            if seg_text and native_transformer.available():
                native = await asyncio.to_thread(
                    native_transformer.generate, seg_text, list(session.history_log))
                if native and not looks_repetitive(native):
                    if trace is not None:
                        trace.append({"stage": "native_causal_transformer",
                                      "segment": seg_text, "surface": native})
                    return native
        except Exception:
            logger.error("native causal transformer failed", exc_info=True)
        # No retrieval, word, generic, template, or character fallback. If the
        # native transformer emits no admitted surface, this turn supplies no
        # fragment; it is not replaced by an agent-authored response mechanism.
        if trace is not None:
            trace.append({"stage": "defer", "segment": seg_text, "surface": ""})
        return ""

    async def _generate_turn_surface(self, final_content, image_chars, session,
                                     ukey, diag, rng=None, trace=None):
        """Run the complete engine-owned conversational generation surface.

        Discord and the read-only end-to-end instrument call this same method:
        utterance segmentation, native causal generation, and fragment
        composition. The historical word/generic fallback is retired. Teacher judgement and correction happen
        later and are deliberately not part of the engine-owned surface.
        """
        segments = segment_utterance(final_content)
        fragments = []
        segment_traces = []
        native_feedback = []
        rag_feedback = False
        for si, seg in enumerate(segments):
            seg_context = ((image_chars if si == 0 else [])
                           + ['\x02'] + list(seg) + ['\x03'])
            stage_trace = []
            frag = await self._generate_fragment_multiscale(
                seg_context, session, ukey, diag, rng=rng, trace=stage_trace)
            segment_traces.append({"segment": seg, "stages": stage_trace,
                                   "accepted": bool(frag)})
            if frag:
                fragments.append(frag)
                if any(stage.get("stage") == "native_causal_transformer"
                       for stage in stage_trace):
                    native_feedback.append((seg, frag))
                if any(stage.get("stage") == "rag_pair_response"
                       for stage in stage_trace):
                    rag_feedback = True

        good = [fragment for fragment in fragments
                if not looks_repetitive(fragment)]
        if good:
            composed = "\n\n".join(good)
            final_stage = "composed_fragments"
        elif fragments:
            composed = fragments[0]
            final_stage = "raw_fragment"
        else:
            composed = ""
            final_stage = "defer"

        if trace is not None:
            trace.append({"segments": segment_traces, "final_stage": final_stage,
                          "surface": composed})
        self._last_native_feedback = native_feedback
        self._last_rag_feedback = rag_feedback
        return composed

    async def on_message(self, message):
        # Ignore own messages
        if message.author == self.user:
            return
            
        ALLOWED_CHANNEL_ID = 1523685773998555227
        if message.channel.id != ALLOWED_CHANNEL_ID:
            return
            
        # Stop auto distillation instantly when interrupted by a real user
        if self.auto_task and not self.auto_task.done():
            self.auto_task.cancel()
            self.auto_task = None
            model_pool.unload_all()
            await message.channel.send("*(User response detected)* **[AUTO MODE OFF]** Background tutoring paused.")
            
        self.last_interaction_time = time.time()
        self.primary_channel = message.channel
        ukey = f"discord_{zlib.crc32(str(message.author.id).encode())}"
        self.primary_ukey = ukey

        # Prevent Unison from babbling in response to unsynced text-based slash commands
        if message.content.startswith('/'):
            await message.channel.send("*(Please use the native Discord Slash Command menu to trigger this. If you do not see the popup when typing '/', press Cmd+R / Ctrl+R to reload your Discord client so it syncs.)*")
            return

        try:
            session = session_manager.get_or_create(ukey)
                
            # Per-Letter Exact Tokenization with Atomic Speaker Demarcation
            diag = GenerationDiagnostics(message.content, ukey)
            
            # ── Multimodal Encoding ──
            # Check for image and audio attachments
            image_chars = []
            audio_text = ""
            
            with Timer() as t_tok:
                for attachment in message.attachments:
                    if attachment.content_type and attachment.content_type.startswith('image/'):
                        try:
                            img_bytes = await attachment.read()
                            encoded = image_encoder.encode(img_bytes)
                            image_chars.extend(encoded)
                            logger.info(f"Image attached: {attachment.filename} → {len(encoded)} chars")
                        except Exception as e:
                            logger.error(f"Failed to encode image {attachment.filename}: {e}")
                    elif attachment.content_type and (attachment.content_type.startswith('audio/') or attachment.content_type.startswith('video/')):
                        try:
                            audio_bytes = await attachment.read()
                            # Whisper transcription
                            if whisper.available:
                                success, text = await asyncio.to_thread(whisper.listen, audio_bytes)
                                if success and text:
                                    audio_text += text + " "
                                    logger.info(f"Audio transcribed: {text}")
                        except Exception as e:
                            logger.error(f"Failed to transcribe audio {attachment.filename}: {e}")
                
                # Combine transcribed text with message content
                final_content = message.content
                if audio_text:
                    final_content = f"{audio_text.strip()} {final_content}".strip()
            diag.tokenization_ms = t_tok.elapsed_ms
            
            # Context before the prompt (used to rewind time on failure)
            context_before_prompt = list(session.working_context)
            
            # Record the user's interaction in the fingerprint database
            user_fingerprints.get_or_create(ukey, display_name=message.author.display_name, discord_id=message.author.id)
            user_fingerprints.record_interaction(ukey, session.session_id)
            
            # Record the user's turn in the session using final_content (includes transcriptions)
            session.add_turn("user", final_content)
            
            # If there were images, also extend with image chars before the text turn
            if image_chars:
                wc = session.working_context
                text_turn_len = len(final_content) + 2  # \x02 + text + \x03
                text_turn = wc[-text_turn_len:]
                session.working_context = wc[:-text_turn_len] + image_chars + text_turn
            
            # Bank the current session state as an orbit
            with Timer() as t_bank:
                omni_memory.hold_orbit(session.working_context, ukey=ukey)
                session.episodic_memory.hold_orbit(session.working_context, ukey=ukey)
            diag.orbit_bank_ms = t_bank.elapsed_ms
            
            # Anchor generation to the CURRENT user turn — NOT the whole accumulated
            # conversation. Generating from the full working_context made every prompt
            # match ~1500 chars of history that all led back to one dominant orbit (the
            # first greeting), so it reproduced that same answer for every question and
            # never looked at what was actually asked. Retrieval is keyed to the current
            # turn; the corrections /auto learns are themselves prompt-keyed, and the
            # teacher still receives the full conversation when tutoring/clarifying.
            #
            # SEGMENTATION: a multi-intent message ("hello, do you remember my name?
            # introduce yourself…") is one long context, so a single walk locks onto
            # one deep orbit and answers only one part (or dumps a stale long orbit).
            # Split the turn into sub-utterances and answer EACH from its own deepest
            # orbit, then compose the fragments. A single-sentence turn yields one
            # segment, so ordinary chats behave exactly as before.
            composed = await self._generate_turn_surface(
                final_content, image_chars, session, ukey, diag)
            generated_chars = list(composed)

            diag.finish(len(generated_chars), composed)

            if generated_chars:
                out_text = "".join(generated_chars)

                # The system speaks for itself. Its own generated output is what is
                # shown — always. The teacher NEVER produces a live user-facing reply
                # in the system's name; its only role is to JUDGE this output (below)
                # and, when it is bad, TUTOR a correction through /auto that is baked
                # into the graph so the system answers from its own memory next time.

                display_text = format_unison_output(out_text)

                # Intercept Tool JSON if present
                tool_result = tools_orchestrator.parse_and_execute(out_text)
                if tool_result:
                    display_text += f"\n\n**🔧 Tool Execution:**\n```\n{tool_result}\n```"
                    out_text += f"\n\n[TOOL_OUTPUT]\n{tool_result}\n[/TOOL_OUTPUT]"
                    
                # Record Unison's response in the session
                session.add_turn("unison", out_text)
                
                # Stamp the token provenance
                proof = unison_identity.generate_response_proof(session.session_id, session.turn_count, out_text)
                
                with Timer() as t_post:
                    omni_memory.hold_orbit(session.working_context, ukey=ukey)
                diag.post_gen_bank_ms = t_post.elapsed_ms
                
                view = FeedbackView(
                    ukey, list(session.working_context), generated_chars, context_before_prompt,
                    user_prompt_text=message.content, response_text=out_text,
                    native_feedback=getattr(self, "_last_native_feedback", []),
                    rag_feedback=getattr(self, "_last_rag_feedback", False)
                )
                
                with Timer() as t_send:
                    msg_obj = await send_chunked(message.channel, display_text, view=view)
                diag.discord_send_ms = t_send.elapsed_ms
                
                # Auto Voice Mode
                if self.voice_mode and kokoro.available:
                    speak_text = clean_tts_text(out_text)
                    
                    if speak_text:
                        success, wav_bytes, dur = await asyncio.to_thread(kokoro.speak, speak_text)
                        if success:
                            wav_file = discord.File(io.BytesIO(wav_bytes), filename="unison_voice.wav")
                            await message.channel.send(f"🔊 *Unison speaks ({dur:.0f}ms):*", file=wav_file)
                
                # Log full diagnostics (always logged to file)
                diag.log()
                
                # ── Live Self-Feedback ──
                if diag.suffix_depths:
                    avg_depth = sum(diag.suffix_depths) / len(diag.suffix_depths)
                    avg_cands = sum(diag.candidate_counts) / len(diag.candidate_counts)
                else:
                    avg_depth, avg_cands = 0, 0
                
                self_rating, self_reason = self.teacher.self_rate(avg_depth, avg_cands, len(generated_chars))

                # Strip thinking and tool outputs for evaluation
                clean_eval_text = out_text
                if '\x05' in clean_eval_text:
                    clean_eval_text = clean_eval_text.split('\x05')[-1].strip()
                if "[TOOL_OUTPUT]" in clean_eval_text:
                    clean_eval_text = clean_eval_text.split("[TOOL_OUTPUT]")[0].strip()

                # ── FOLD-NATIVE CRITIC (coherence_value.ep) — the engine's OWN value, no
                # Gemma: the reply's content words against each other (word<->statement)
                # and against the conversation (statement<->context), read at the lock 1/2.
                resp_content = _content_words(tokenize(clean_eval_text))
                conv_words = tokenize(" ".join(t for _, t in session.turns[-6:]))
                word_scale, ctx_scale = word_engine.coherence_score(resp_content, conv_words)
                fold_coherent = (word_scale >= 0.5)
                fold_rating = "good" if fold_coherent else "bad"

                # ── COMPETITIVE LADDER (live): head-to-head, sovereign when graduated ──
                territory = "general"
                sovereign = self.grad_ledger.has_graduated(territory)
                if sovereign:
                    # Sovereign regime: the engine judges ITSELF by the fold critic — no
                    # teacher round-trip (own-teacher / observer). The teacher has been beaten.
                    teacher_rating = fold_rating
                    teacher_reason = f"fold-critic (sovereign): word_scale={word_scale:.2f}"
                else:
                    eval_curriculum = "sft_tool_use" if tool_result else "general"
                    history_str = ""
                    prev_turns = list(session.history_log)   # reliable append-only source
                    if prev_turns:
                        history_str = "Recent conversation history:\n"
                        for role, text in prev_turns:
                            name = "User" if role == "user" else "Unison"
                            history_str += f"{name}: {text}\n"
                    teacher_rating, teacher_reason = await asyncio.to_thread(
                        self.teacher.rate, message.content, clean_eval_text, curriculum=eval_curriculum, history=history_str
                    )

                # Record the match: the engine WINS when its own coherence critic holds AND
                # (when not yet sovereign) the teacher agrees — it stood up head-to-head.
                omni_won = fold_coherent and (sovereign or teacher_rating == "good")
                self.grad_ledger.record_match(territory, omni_won)
                if self.grad_ledger.has_graduated(territory) and territory not in self._graduated_announced:
                    self._graduated_announced.add(territory)
                    w, l = self.grad_ledger.scores[territory]
                    await message.channel.send(
                        f"*(Unison graduated **{territory}** — p = {w}/{w+l} ≥ 1/2, the fold lock. "
                        f"It is now sovereign here: it judges its own coherence, no teacher.)*")

                # Track the growth trajectory + log the full turn for live monitoring.
                self.recent_coherence.append(word_scale)
                self.turn_counter += 1
                _gs = self.grad_ledger.scores.get("general", [0, 0])
                _gp = _gs[0] / (_gs[0] + _gs[1]) if (_gs[0] + _gs[1]) else 0.0
                chat_logger.info(
                    f"[TURN {self.turn_counter}] user={message.content!r} | reply={clean_eval_text[:220]!r} | "
                    f"self={self_rating} teacher={teacher_rating} fold_coherence={word_scale:.2f} ctx={ctx_scale:.2f} "
                    f"won={omni_won} sovereign={sovereign} graduation_p={_gp:.2f}")

                agree = (self_rating == teacher_rating)
                agree_emoji = "✅" if agree else "⚠️"
                
                auto_severed = False
                corrected_live = False
                if self_rating == "bad" or teacher_rating == "bad":
                    # Remove the bad trajectory from the graph.
                    omni_memory.prune_orbit(session.working_context, ukey=ukey)
                    # Reward observations are deposited only in the learning
                    # system that produced the served surface.
                    if getattr(self, "_last_rag_feedback", False):
                        await asyncio.to_thread(pair_retrieval.mark_feedback, False)
                    for native_prompt, native_surface in getattr(
                            self, "_last_native_feedback", []):
                        await asyncio.to_thread(
                            native_transformer.mark_feedback,
                            native_prompt, native_surface, False)
                    reason = teacher_reason if teacher_rating == "bad" else self_reason

                    # ── ON-THE-SPOT TEACHING (don't leave the turn dead) ──
                    # Get the correct in-persona answer NOW — the correction prompt
                    # carries the FULL conversation so the teacher isn't confused about
                    # the discussion — recover the turn live as Unison, and teach the
                    # corrected exchange immediately so it answers from its own memory
                    # next time (no waiting for /auto).
                    correction = build_correction_prompt(session, final_content, clean_eval_text, reason=reason)
                    instruction = correction.replace("[CORRECTION]", "").strip()
                    teacher_ans = ""
                    try:
                        teacher_ans = await asyncio.to_thread(self.teacher.ask, instruction)
                    except Exception:
                        logger.error("live correction: teacher.ask failed", exc_info=True)

                    if teacher_ans and teacher_ans.strip():
                        corrected_orbit = (['\x02'] + list(final_content) + ['\x03']
                                           + list(teacher_ans) + ['\x02'])
                        omni_memory.hold_orbit(corrected_orbit, ukey=ukey)
                        session.episodic_memory.hold_orbit(corrected_orbit, ukey=ukey)

                        # Replace the bad turn with the correction so the conversation
                        # stays coherent for the long horizon (not rolled back dead).
                        if session.turns and session.turns[-1][0] == "unison":
                            session.turns[-1] = ("unison", teacher_ans)
                        session.working_context = (list(context_before_prompt)
                                                   + ['\x02'] + list(final_content) + ['\x03']
                                                   + list(teacher_ans) + ['\x02'])

                        recovered = format_unison_output(teacher_ans)
                        await send_chunked(
                            message.channel,
                            "*(oops — sorry about that, I'm still learning)*\n\n" + recovered)
                        log_learning_event("live_correction_taught", final_content, teacher_ans,
                                           f"reason={reason}", teacher_ans, ukey=ukey)
                        # The correction becomes held LEARNING MATERIAL: multiple expressions
                        # of one meaning (generation_selection_law.ep — re-expression requires
                        # >= b = 2 held expressions; a second phrasing is requested so the
                        # taught meaning can serve re-expressed, never verbatim).
                        _ans = (teacher_ans.split('\x05')[-1].strip()
                                if '\x05' in teacher_ans else teacher_ans.strip())
                        if _ans:
                            _v = ""
                            try:
                                _v2 = await asyncio.to_thread(
                                    self.teacher.ask,
                                    f"The user says: {final_content!r}. Respond as Unison — "
                                    f"natural, warm, in your own voice, in 1-2 short sentences, "
                                    f"worded DIFFERENTLY from: {_ans!r}")
                                _v = (_v2.split('\x05')[-1].strip()
                                      if '\x05' in _v2 else _v2.strip())
                            except Exception:
                                logger.error("second phrasing failed", exc_info=True)
                            await asyncio.to_thread(
                                pair_retrieval.add_taught, final_content, _ans, ukey,
                                [v for v in (_v,) if v and v != _ans])
                        corrected_live = True
                    else:
                        # Teacher unavailable — fall back to deferred /auto + rollback.
                        omni_ledger.add_prompt(ukey, correction, original_prompt=final_content)
                        session.working_context = list(context_before_prompt)
                        if len(session.turns) >= 2:
                            session.turns = session.turns[:-2]
                            session.turn_count = len(session.turns)
                        auto_severed = True
                else:
                    omni_ledger.add_prompt(ukey, final_content)
                    if teacher_rating == "good":
                        omni_memory.fold_orbit(session.working_context, ukey=ukey)
                        if getattr(self, "_last_rag_feedback", False):
                            await asyncio.to_thread(pair_retrieval.mark_feedback, True)
                        for native_prompt, native_surface in getattr(
                                self, "_last_native_feedback", []):
                            await asyncio.to_thread(
                                native_transformer.mark_feedback,
                                native_prompt, native_surface, True)
                
                # RECORD the finalized turn into the append-only history_log so the NEXT
                # turn's teacher sees the real conversation (e.g. the user's name). The reply
                # shown was the live correction if one happened, else the engine's own reply.
                if corrected_live and teacher_ans:
                    shown_reply = teacher_ans.split('\x05')[-1].strip() if '\x05' in teacher_ans else teacher_ans.strip()
                else:
                    shown_reply = clean_eval_text
                session.record_exchange(final_content, shown_reply)

                log_learning_event("live_self_feedback", message.content, out_text,
                    f"self={self_rating}|teacher={teacher_rating}|agree={agree}",
                    f"self_reason={self_reason}|teacher_reason={teacher_reason}",
                    ukey=ukey)
                
                # Post diagnostics to Discord only if diagnostic mode is ON
                if self.diagnostic_mode:
                    session_info = f"Session: {session.session_id} | Turns: {session.turn_count} | WM: {len(session.working_context)} chars"
                    feedback_line = f"**[Self-Feedback]** {agree_emoji} Self={self_rating} | Teacher={teacher_rating} — *{teacher_reason}*"
                    if corrected_live:
                        feedback_line += "\n*(Corrected on the spot: pruned the bad trajectory, taught the right answer, and recovered the turn live)*"
                    elif auto_severed:
                        feedback_line += "\n*(Auto-Severed bad trajectory from Synaptic Graph and recorded in Active Ledger)*"
                    await message.channel.send(diag.discord_summary() + "\n" + session_info + "\n" + feedback_line)
                
        except Exception as e:
            logger.error("Error processing message", exc_info=True)
            await message.channel.send(f"**[Error]:** {str(e)}")

def init_subsystems():
    global omni_memory, session_manager, image_encoder, audio_encoder, unison_identity
    global user_fingerprints, observer, model_pool, flux_gen, kokoro, whisper, tools_orchestrator
    
    # Runtime-arm selection is explicit. With no environment setting the live
    # legacy surface is unchanged.
    fluency_arm = os.environ.get("UNISON_FLUENCY_ARM")
    if fluency_arm and not word_engine.fluency_identity().get("runtime_arm"):
        identity = word_engine.configure_registered_fluency(fluency_arm)
        logger.info(
            "Registered response-fluency arm active: %s (%s)",
            identity["runtime_arm"]["path"], identity["sha256"])

    # Initialize only if not already initialized
    if omni_memory is not None: return
    
    omni_memory = SynapticGraph()
    session_manager = SessionManager()
    image_encoder = ImageEncoder(grid_size=8)
    audio_encoder = AudioEncoder()
    unison_identity = UnisonIdentity()
    user_fingerprints = UserFingerprint()
    observer = ObserverTeacher()
    model_pool = ModelPool()
    flux_gen = FluxGenerator()
    kokoro = KokoroSpeaker()
    whisper = WhisperListener()
    tools_orchestrator = ToolOrchestrator()

if __name__ == "__main__":
    import atexit
    import signal
    
    init_subsystems()
    
    intents = discord.Intents.default()
    intents.message_content = True
    client = SFTDiscordClient(intents=intents)
    
    # Ensure sessions are banked and graph is saved on any shutdown
    def _shutdown_save():
        logger.info("Shutdown: banking all active sessions...")
        for ukey, full_seq in session_manager.end_all():
            if full_seq and len(full_seq) > 10:
                omni_memory.hold_orbit(full_seq, ukey=ukey)
                logger.info(f"  Banked session for {ukey}: {len(full_seq)} chars")
        logger.info("Shutdown: force-saving graph memory...")
        omni_memory.force_save()
        logger.info("Shutdown: saving learned couplings + span quality...")
        try:
            word_engine.save_couplings()
            word_engine.save_span_quality()
        except Exception:
            logger.error("learning save on shutdown failed", exc_info=True)
    
    atexit.register(_shutdown_save)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    
    # Load from .env file
    token = os.environ.get("DISCORD_TOKEN")
    if not token and os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("DISCORD_TOKEN="):
                    token = line.strip().split("=", 1)[1]
                    break
    # The project benchmark tooling already uses this user-private token file.
    # Accept the same 0600-protected source for the live launcher so a clean
    # restart does not require copying the secret into the repository.
    private_token_path = os.path.expanduser("~/.unison_discord_token")
    if not token and os.path.isfile(private_token_path):
        with open(private_token_path, "r") as handle:
            token = handle.read().strip()
                    
    if token:
        logger.info("Token loaded successfully. Connecting to Discord...")
        client.run(token)
    else:
        logger.error("DISCORD_TOKEN not found in environment or .env file.")
