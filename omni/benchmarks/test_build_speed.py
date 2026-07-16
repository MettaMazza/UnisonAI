import time, zlib, collections
def bkey(tup): return (zlib.crc32(" ".join(tup).encode()) % 1000003,)
chars = ['a'] * 100000
stores = [collections.defaultdict(lambda: collections.defaultdict(int)) for _ in range(7)]
t0 = time.time()
for i in range(len(chars)-1):
    nxt = chars[i+1]
    for L in range(7):
        if i-L+1 < 0: break
        stores[L][bkey(tuple(t.lower() for t in chars[i-L+1:i+1]))][nxt] += 1
print(f"Time for 100k: {time.time()-t0:.2f}s")
