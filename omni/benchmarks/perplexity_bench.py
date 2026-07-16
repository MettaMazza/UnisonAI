import math
import sys
import importlib.util

BASE = "/Users/mettamazza/Desktop/Smithian Fold Theory"
ENGINE = BASE + "/fold_ai/unison_chat.py"

def calculate_perplexity(model, text):
    log_prob_sum = 0
    count = 0
    ctx = []
    
    for char in text:
        dist = model.mixed_dist(ctx)
        total = sum(dist.values()) if dist else 0
        
        if total > 0 and char in dist:
            prob = float(dist[char]) / float(total)
        else:
            prob = 1e-5  # Smoothing floor
            
        log_prob_sum += math.log2(prob)
        count += 1
        
        ctx.append(char)
        if len(ctx) > model.CTX_MAX:
            ctx.pop(0)
            
    avg_log_prob = log_prob_sum / count
    return 2 ** (-avg_log_prob), -avg_log_prob  # perplexity, bits per byte

def main():
    print("Loading UnisonAI store...", flush=True)
    src = open(ENGINE).read()
    spec = importlib.util.spec_from_loader("unison_engine", loader=None, origin=ENGINE)
    U = importlib.util.module_from_spec(spec)
    U.__file__ = ENGINE
    sys.modules["unison_engine"] = U
    exec(compile(src, ENGINE, "exec"), U.__dict__)
    
    test_text = "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversation?'"
    
    print(f"Testing perplexity on {len(test_text)} characters...", flush=True)
    ppl, bpb = calculate_perplexity(U, test_text)
    print(f"Results:\nBits Per Byte (Cross-Entropy Loss): {bpb:.4f}\nCharacter Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()
