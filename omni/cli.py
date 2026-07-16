import sys
from omni.core import fwht_2d, ifwht_2d
from omni.memory import SynapticGraph, predict_next
from omni.teacher_scaffold import LocalTeacher, GraduationLedger

def text_to_dyadic(text):
    block = [[0]*64 for _ in range(64)]
    for i, char in enumerate(text[:64*64]):
        row = i // 64
        col = i % 64
        block[row][col] = ord(char)
    transformed = fwht_2d(block)
    flat = [val for r in transformed for val in r]
    return tuple(flat[:32])

def dyadic_to_text(signature):
    block = [[0]*64 for _ in range(64)]
    flat = list(signature) + [0] * (64*64 - 32)
    for i in range(64*64):
        block[i//64][i%64] = flat[i]
    recovered = ifwht_2d(block)
    chars = []
    for row in recovered:
        for val in row:
            if val > 0 and val < 128:
                chars.append(chr(val))
    return "".join(chars).strip()

def run_cli():
    print("======================================================")
    print(" SFT OMNI ARCHITECTURE - LIVE LOCAL TERMINAL")
    print("======================================================")
    print("Engine: Zero Parameters, Exact Fractions.")
    print("Scaffold: qwen3.6-27b:latest (via Ollama)")
    print("Type your message below. Type 'exit' to quit.\n")
    
    graph = SynapticGraph()
    teacher = LocalTeacher()
    ledger = GraduationLedger()
    
    ukey = "local_user_1"
    context = []
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            # COMPOSE from the FOUNDATION (never verbatim recall) — the same standard as
            # the live bot: retrieve+recombine coherent on-topic spans, fallback to the
            # word-tier unfold. The engine never replays memory char-by-char.
            from omni.word_engine import word_engine, tokenize, _content_words
            import random as _r
            word_engine.ensure_built(graph, ukey)
            schema = _content_words(tokenize(user_input))
            resp = word_engine.retrieve_and_compose(schema, _r.Random())
            if not resp:
                resp = word_engine.unfold_response(schema, _r.Random())
            if resp:
                print(f"Omni Engine: {resp}")
            else:
                # Only when the engine composes nothing does the teacher scaffold answer.
                teacher_ans = teacher.ask(user_input)
                print(f"Teacher Scaffold: {teacher_ans}")
                
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    run_cli()
