import time, zlib
from collections import defaultdict
STORE_BOUND = 1000003

def _key_fast(t):
    if STORE_BOUND:
        return (zlib.crc32(" ".join(t).encode()) % STORE_BOUND,)
    return tuple(t)

text = "This is a sample text for testing write_orbits. " * 10000
t0 = time.time()
stores = [defaultdict(lambda: defaultdict(int)) for _ in range(7)]
tl = list(text)
tl_lower = [x.lower() for x in tl]
for i in range(len(tl) - 1):
    nxt = tl[i + 1]
    for L in range(0, 7):
        if i - L + 1 < 0: break
        stores[L][_key_fast(tl_lower[i - L + 1:i + 1])][nxt] += 1
print(f"Time taken: {time.time()-t0:.2f}s for {len(text)} characters")
