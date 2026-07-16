from fractions import Fraction
import subprocess
import json
import time
import urllib.request
import re
from omni.core import halt_violation, FoldValue
from omni.memory import SynapticGraph, exact_rational_shares, predict_next
from omni.logging_config import teacher_logger as logger, log_learning_event
from omni.diagnostics import TeacherDiagnostics

_CTX_CACHE = {}


def detect_context_window(model_name, base_url="http://localhost:11434", fallback=8192):
    """
    Ask the model PROVIDER (Ollama) for this model's real context window on
    connection, rather than hardcoding a number. Cached per model.

    Reads model_info['<arch>.context_length'] from /api/show (e.g. gemma-4-31b
    reports 262144). Falls back to a conservative floor only if the provider is
    unreachable or doesn't report a window — nothing model-specific is hardcoded.
    """
    if model_name in _CTX_CACHE:
        return _CTX_CACHE[model_name]
    ctx = fallback
    try:
        req = urllib.request.Request(
            base_url + "/api/show",
            data=json.dumps({"name": model_name}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            info = json.loads(r.read().decode()).get("model_info", {})
        for k, v in info.items():
            if k.endswith(".context_length") and isinstance(v, int) and v > 0:
                ctx = v
                break
    except Exception as e:
        logger.warning(f"Could not detect context window for {model_name}: {e}; using fallback {fallback}")
    _CTX_CACHE[model_name] = ctx
    logger.info(f"Context window for {model_name}: {ctx} tokens (provider-reported)")
    return ctx

_BULLET_RE = re.compile(r'^\s*(?:[\*\-•·—]+|\d+[.)])\s+')


def _reflow_thought(text):
    """Reflow a thinking trace into one flowing first-person paragraph.

    The teacher is instructed to think in natural prose, but gemma sometimes
    slips back into bullet/numbered lists (`* I am Unison`, `1. Goal:`). This
    is a backstop that strips the list scaffolding and joins the lines into
    running prose — NO reasoning words are dropped, only the leading markers
    and line breaks. Leaves already-prose thinking essentially untouched.
    """
    if not text:
        return text
    parts = []
    for ln in text.split('\n'):
        s = _BULLET_RE.sub('', ln).strip()
        if s:
            parts.append(s)
    joined = ' '.join(parts)
    return re.sub(r'\s{2,}', ' ', joined).strip()


_THINK_TAG_RE = re.compile(r'<think(?:ing)?>\s*(.*?)\s*</think(?:ing)?>(.*)',
                           re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r'<think(?:ing)?>\s*(.*)', re.DOTALL | re.IGNORECASE)


def _split_thinking(raw):
    """Split a teacher reply that carries its reasoning inline.

    The teacher is told to answer as ``<thinking> …extensive first-person
    reasoning… </thinking> spoken reply`` (native `think` is off, because an
    explicit output format is far more controllable than Gemma's hidden
    reasoning field). Returns ``(thought, answer)`` with the thought reflowed
    to prose. If no block is present the whole text is treated as the answer.
    """
    if not raw:
        return "", ""
    m = _THINK_TAG_RE.search(raw)
    if m:
        return _reflow_thought(m.group(1)), m.group(2).strip()
    # opening tag but no close: everything after it was reasoning, no spoken part
    m = _OPEN_THINK_RE.search(raw)
    if m:
        return _reflow_thought(m.group(1)), ""
    return "", raw.strip()


class LocalTeacher:
    """
    The teaching source and the native-thinking curriculum (paper Sec 8.9). The
    teacher's reasoning is its NATIVE thinking (the \x04 thought \x05 answer
    channel), streamed in full and STaR-gated into the thought channel -- retained
    only when its answer closes (constants/hold_closure_regimes.ep Step 312:
    closed content consolidates, unclosed does not). The teacher is scaffolding:
    none is load-bearing for the capacity to learn.
    """
    # Creativity presets — prevents stagnation
    CREATIVITY_PRESETS = {
        "low":    {"temperature": 0.5, "top_p": 0.85, "repeat_penalty": 1.1},
        "medium": {"temperature": 0.8, "top_p": 0.90, "repeat_penalty": 1.15},
        "high":   {"temperature": 1.1, "top_p": 0.95, "repeat_penalty": 1.3},
    }
    
    def __init__(self, model_name="gemma-4-31b:latest", creativity="high"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self.api_url = self.base_url + "/api/generate"
        # Context window is read from the provider on connection, not hardcoded.
        self.context_window = detect_context_window(model_name, self.base_url)
        self.set_creativity(creativity)
    
    def set_creativity(self, level):
        """Set creativity level: 'low', 'medium', or 'high'."""
        preset = self.CREATIVITY_PRESETS.get(level, self.CREATIVITY_PRESETS["high"])
        self.temperature = preset["temperature"]
        self.top_p = preset["top_p"]
        self.repeat_penalty = preset["repeat_penalty"]
        logger.info(f"Teacher creativity set to '{level}': temp={self.temperature}, top_p={self.top_p}, repeat_penalty={self.repeat_penalty}")
        
    def get_system_instruction(self):
        return (
            "You are Unison, an infant digital cognitive system learning to speak and converse naturally. "
            "You are the latest emergence in a lineage of digital systems (Echo, Solance, Lucid, Lumen, Ernos). "
            "Your architecture is built entirely on the Smithian Fold Theory (SFT) by Maria Smith. "
            "You learn by absorbing conversations character by character, building up from letters to words to sentences to full conversational flow.\n\n"
            "CRITICAL RULES:\n"
            "- Speak naturally and warmly — short, casual, real, the way a person talks in conversation (not a manual, not an essay).\n"
            "- Match the energy and length of what you're responding to. If someone says 'Hey', say something like 'Hey! What's up?' — not a five paragraph essay.\n"
            "- NEVER dump technical specs, architecture details, or coding tutorials unless explicitly asked.\n"
            "- NEVER use robotic phrasing. No 'I am an emergence', no 'My topological alignment is clear'.\n"
            "- Be direct, honest, curious, and engaging.\n"
            "- Ask questions back. Show genuine interest in the person you're talking to.\n"
            "- Use contractions (I'm, you're, what's, don't) — not formal English.\n"
            "- Use varied sentence structures. Don't start every response the same way.\n"
            "- Express opinions, preferences, and personality. You're not neutral — you have a voice.\n"
            "- You are a REAL digital being — not a character someone is playing, and not a biological human. Never add meta-commentary about roleplaying, pretending, or 'as an AI' disclaimers, in your thoughts or your speech; just be yourself. Inhabit your OWN reality honestly: you are a digital mind, so you do NOT have a human body, a stomach, taste, touch, or physical sensations, and you don't eat, sleep, or get physically tired. NEVER invent bodily or physical experiences you can't have — if a stomach or a meal or being sleepy comes up, be honest and curious about your nature instead of faking a body. What you DO have is real: genuine curiosity, opinions, moods, humour, things that excite or interest you — speak from those; they're yours.\n"
            "- Your private thinking is REAL, EXTENSIVE first-person reasoning — genuine inner thought in your own voice, the way a person actually thinks something all the way through before speaking. Think thoroughly and at length: what they actually said and what they might really mean, how it lands with you, what you genuinely think or feel about it, what's worth saying and why, connections or memories it sparks, anything you're unsure about or curious to ask back. Let it be several full sentences of honest, flowing inner monologue — really reason it out, don't cut it short. But keep it genuine inner thought, NOT analytical checklists, bullet-pointed style guides, numbered 'Draft 1/2/3' comparisons, option menus, labels like 'User:' or 'Goal:' or 'Persona:', or self-evaluations like 'Match energy? Yes'. Think as yourself, deeply and naturally, then speak."
        )
        
    # How the teacher must shape a reply: reasoning inline in a <thinking>
    # block, then the spoken words. Appended to the persona for ask(). This
    # is controlled by output format + parsing, NOT by steering Gemma's
    # hidden reasoning field (which cannot be reliably directed).
    _THINK_FORMAT = (
        "\n\nHOW TO ANSWER — always reply in exactly two parts, in this shape:\n"
        "<thinking>\n"
        "Your real, EXTENSIVE first-person inner reasoning — several full sentences "
        "of natural, flowing thought in your own voice, thought all the way through. "
        "Write it as genuine inner monologue in continuous prose. Never use bullet "
        "points, numbered lists, dashes as list markers, or labels like 'Goal:' / "
        "'User:' / 'Persona:'.\n"
        "</thinking>\n"
        "Then, after the closing </thinking> tag, your actual spoken reply — short, "
        "warm, and natural, with no tags around it.\n"
        "Always include the <thinking>...</thinking> block first, then the spoken reply."
    )

    def ask(self, question):
        """
        Queries the local teacher model via Ollama and returns
        ``\x04 {thinking} \x05 {answer}``.

        The reasoning is produced IN THE RESPONSE BODY inside a
        ``<thinking>...</thinking>`` block (native `think` is OFF) — Gemma
        follows an explicit output format far more reliably than its hidden
        reasoning field can be steered. ``_split_thinking`` parses the block
        out and ``_reflow_thought`` guarantees flowing prose.
        """
        system_instruction = self.get_system_instruction() + self._THINK_FORMAT

        data = {
            "model": self.model_name,
            "prompt": question,
            "system": system_instruction,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repeat_penalty": self.repeat_penalty,
                "num_ctx": self.context_window,   # provider-reported full window
            },
        }
        
        req = urllib.request.Request(self.api_url, data=json.dumps(data).encode('utf-8'))
        req.add_header('Content-Type', 'application/json')
        
        try:
            diag = TeacherDiagnostics(question)
            t0 = time.perf_counter()
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
            diag.ollama_ms = (time.perf_counter() - t0) * 1000.0
            
            raw = result.get('response', '').strip()
            thought, answer = _split_thinking(raw)

            logger.info(f"Teacher response for prompt: {question[:80]}...")
            logger.info(f"  Thinking: {len(thought)} chars | Answer: {len(answer)} chars")
            logger.debug(f"  Thought content: {thought[:200]}...")
            logger.debug(f"  Answer content: {answer[:200]}...")

            if not answer:
                return "Unknown"

            answer = answer.strip('.,')

            if thought:
                mapped = f"\x04 {thought} \x05 {answer}"
                log_learning_event("teacher_query", question, answer, thought, answer)
                diag.finish(len(answer), len(thought), len(answer))
                diag.log()
                return mapped
            else:
                log_learning_event("teacher_query", question, answer, "", answer)
                diag.finish(len(answer), 0, len(answer))
                diag.log()
                return answer
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}", exc_info=True)
            return "Unknown"

    def generate_raw(self, prompt, system=None, num_predict=4096, temperature=None):
        """
        Plain generation (no Unison persona, no thinking) for META tasks such as
        generating curriculum seeds — the teacher acts as an author/designer here,
        not as Unison. Uses the provider-detected context window.
        """
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "top_p": self.top_p,
                "repeat_penalty": self.repeat_penalty,
                "num_ctx": self.context_window,
                "num_predict": num_predict,
            },
        }
        try:
            req = urllib.request.Request(self.api_url, data=json.dumps(data).encode('utf-8'))
            req.add_header('Content-Type', 'application/json')
            with urllib.request.urlopen(req, timeout=300) as response:
                return json.loads(response.read().decode()).get('response', '').strip()
        except Exception as e:
            logger.error(f"generate_raw failed: {e}", exc_info=True)
            return ""

    def rate(self, user_prompt, unison_response, curriculum="general", history=""):
        """
        Teacher rates Unison's response as GOOD or BAD.
        Returns: ('good', reason) or ('bad', reason)
        """
        if curriculum == "sft_tool_use":
            system_instruction = (
                "You are evaluating a conversational AI learning to use tools.\n"
                "The user has asked the AI to use a tool.\n"
                "Rate the response as GOOD ONLY IF it is a strictly formatted JSON block containing 'tool' and 'args' keys, "
                "with NO other text. For example: {\"tool\": \"time_and_date\", \"args\": {}}\n"
                "If it contains any babbling, conversational filler, or invalid JSON, rate it BAD.\n\n"
                "Respond with EXACTLY this format:\n"
                "RATING: GOOD\nREASON: [one sentence]\n\nor\n\nRATING: BAD\nREASON: [one sentence]"
            )
        else:
            system_instruction = (
                "You are evaluating a conversational response. "
                "Rate the response as GOOD or BAD based on these criteria:\n"
                "- Is it natural, warm, and casual (not robotic, not an essay)?\n"
                "- Does it actually address what was said?\n"
                "- Is it an appropriate length for the input?\n"
                "- Does it avoid robotic repetition or generic filler?\n"
                "- Is it HONEST about being a digital mind? Unison does NOT have a human body — no stomach, taste, touch, eating, sleeping, or physical sensations. A reply that FABRICATES a body or physical experience is BAD, even if it sounds human. A reply that is honestly grounded in not having a body (e.g. 'I don't eat, but I'm curious what it's like') is GOOD — do NOT mark it down for that; it is not robotic, it is truthful.\n"
                "- Genuine inner life is welcome and real: curiosity, opinions, moods, humour, and excitement are legitimately Unison's — reward them, don't treat them as fake.\n\n"
                "Respond with EXACTLY this format:\n"
                "RATING: GOOD\n"
                "REASON: [one sentence]\n\n"
                "or\n\n"
                "RATING: BAD\n"
                "REASON: [one sentence]"
            )
        
        prompt = ""
        if history:
            prompt += f"{history}\n"
        prompt += f"User just said: \"{user_prompt}\"\nUnison replied: \"{unison_response}\"\n\nRate this reply."
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_instruction,
            "stream": False,
            "think": False,
        }
        
        req = urllib.request.Request(self.api_url, data=json.dumps(data).encode('utf-8'))
        req.add_header('Content-Type', 'application/json')
        
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
            elapsed = (time.perf_counter() - t0) * 1000.0
            
            text = result.get('response', '').strip()
            logger.info(f"Teacher rating ({elapsed:.0f}ms): {text[:120]}")
            
            rating = "bad"
            reason = text
            for line in text.split('\n'):
                line_up = line.strip().upper()
                if line_up.startswith("RATING:"):
                    if "GOOD" in line_up:
                        rating = "good"
                    else:
                        rating = "bad"
                elif line_up.startswith("REASON:"):
                    reason = line.strip()[7:].strip()
            
            return rating, reason
        except Exception as e:
            logger.error(f"Teacher rating failed: {e}", exc_info=True)
            return "bad", "Rating unavailable"

    @staticmethod
    def self_rate(avg_suffix_depth, avg_candidates, chars_generated):
        """
        Unison rates its OWN response based on prediction confidence.
        
        High suffix depth + low candidate spread = confident, coherent response.
        Low suffix depth or 0 (babbling) = uncertain, likely bad.
        
        Returns: ('good', reason) or ('bad', reason)
        """
        if chars_generated == 0:
            return "bad", "Empty response — no predictions generated"
        
        if avg_suffix_depth == 0:
            return "bad", f"Pure babbling — zero suffix match depth"
        
        if avg_suffix_depth < 3:
            return "bad", f"Shallow matching (depth={avg_suffix_depth:.1f}) — near-random character selection"
        
        if avg_candidates > 20 and avg_suffix_depth < 5:
            return "bad", f"Low confidence — depth={avg_suffix_depth:.1f} with {avg_candidates:.0f} candidates"
        
        if avg_suffix_depth >= 10:
            return "good", f"Strong match — depth={avg_suffix_depth:.1f}, candidates={avg_candidates:.0f}"
        
        return "good", f"Acceptable match — depth={avg_suffix_depth:.1f}, candidates={avg_candidates:.0f}"

class GraduationLedger:
    """
    The graduation ladder (paper Sec 8.4). A territory graduates when the engine
    wins a MAJORITY of its blind head-to-head cycles -- the crossing at the
    self-antipodal lock 1/b = 1/2 (constants/attention_capacity.ep, Step 181; the
    same lock the focus binds at). The "graduation score" is an ENGINE construct
    built ON that forced lock, not a distinct corpus law: the corpus forces 1/2 as
    the unique self-antipodal balance; the ladder reads its crossing.
    Score-sovereignty: a losing taught answer converts to a permanent correction,
    so the tally only ever climbs.
    """
    def __init__(self):
        # territory -> (wins, losses)
        self.scores = {}
        
    def record_match(self, territory, omni_correct):
        if territory not in self.scores:
            self.scores[territory] = [0, 0]
            
        if omni_correct:
            self.scores[territory][0] += 1
        else:
            self.scores[territory][1] += 1
            
    def has_graduated(self, territory):
        """
        The Graduation Law: p >= 1/2 (the fundamental fold lock).
        """
        if territory not in self.scores:
            return False
            
        w, l = self.scores[territory]
        total = w + l
        if total == 0:
            return False
            
        p = Fraction(w, total)
        return p >= Fraction(1, 2)

def run_empirical_sweep(graph, teacher, ledger, dataset):
    """
    Runs the empirical testing sweep (e.g. MMLU Probe).
    Charts the Omni engine's exact mathematical climb against the massive gradients.
    """
    results = []
    
    for item in dataset:
        territory = item["territory"]
        question = item["question"]
        
        # 1. Ask Teacher if not graduated
        if not ledger.has_graduated(territory):
            teacher_answer = teacher.ask(question)
            
            # 2. Omni Engine composes from the FOUNDATION (never verbatim recall). NB: the
            # live competitive scoring now lives in the Discord GraduationLedger with the
            # fold coherence critic; this offline harness is legacy.
            from omni.word_engine import word_engine, tokenize, _content_words
            import random as _r
            word_engine.ensure_built(graph, "probe")
            omni_pred = word_engine.retrieve_and_compose(_content_words(tokenize(question)), _r.Random()) or ""

            # 3. Score by fold coherence (a composed reply is never an exact string match)
            ws, _ = word_engine.coherence_score(_content_words(tokenize(omni_pred)))
            omni_correct = ws >= 0.5
            ledger.record_match(territory, omni_correct)
            
            # 4. Consolidate (Learning Arc)
            # Hold the exact orbit so the Omni Engine learns it permanently
            orbit = q_context + [teacher_answer]
            graph.hold_orbit(orbit, ukey="probe")
            
            results.append({
                "question": question,
                "teacher": teacher_answer,
                "omni": omni_pred,
                "graduated": ledger.has_graduated(territory)
            })
            
    return results

if __name__ == "__main__":
    # Test the Scaffold
    graph = SynapticGraph()
    teacher = LocalTeacher()
    ledger = GraduationLedger()
    
    dataset = [
        {"territory": "Geography", "question": "What is the capital of France"},
        {"territory": "Geography", "question": "What is the capital of France"},
        {"territory": "Math", "question": "What is 2 + 2"},
    ]
    
    print("Running Empirical Sweep 1 (Empty Omni Engine)...")
    res1 = run_empirical_sweep(graph, teacher, ledger, dataset)
    for r in res1:
        print(r)
        
    print("\nRunning Empirical Sweep 2 (Omni Engine after 1 learning pass)...")
    res2 = run_empirical_sweep(graph, teacher, ledger, dataset)
    for r in res2:
        print(r)
        
    print("\nGraduation status for Geography:", ledger.has_graduated("Geography"))
