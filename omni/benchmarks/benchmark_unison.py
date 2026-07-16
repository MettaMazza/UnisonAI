#!/usr/bin/env python3
"""
UnisonAI MMLU Benchmark Suite
Runs the 128-item MMLU probe on UnisonAI (own channels, no external teacher).
"""
import sys
import os
import time
import importlib.util

BASE = "/Users/mettamazza/Desktop/Smithian Fold Theory"
ENGINE = BASE + "/fold_ai/unison_chat.py"

def main():
    print("=== UnisonAI Benchmark Suite ===", flush=True)
    
    # 1. Load engine read-only with redirected logs
    src = open(ENGINE).read()
    OLD = 'LOGFILE = LOGDIR + "/unison.log"'
    NEW = 'LOGFILE = LOGDIR + "/benchmark_unison.log"'
    assert src.count(OLD) == 1, "LOGFILE anchor drifted"
    src = src.replace(OLD, NEW)
    
    spec = importlib.util.spec_from_loader("unison_engine", loader=None, origin=ENGINE)
    U = importlib.util.module_from_spec(spec)
    U.__file__ = ENGINE
    sys.modules["unison_engine"] = U
    
    print("Loading UnisonAI store...", flush=True)
    exec(compile(src, ENGINE, "exec"), U.__dict__)
    
    orbits = sum(len(s) for s in U.stores)
    print(f"Engine loaded successfully: {orbits} orbits, {len(U.SENTS)} sentences.", flush=True)
    
    # 2. Run MMLU sota benchmark for unison-own
    n = 128
    print(f"\nRunning 128-item MMLU probe on UnisonAI...", flush=True)
    t0 = time.time()
    
    # We pass empty models list to only benchmark unison-own (zero parameters)
    result = U.run_sota_bench(models=[], n=n)
    elapsed = time.time() - t0
    
    print(f"\nBenchmark completed in {elapsed:.1f}s.")
    print(result)

if __name__ == "__main__":
    main()
