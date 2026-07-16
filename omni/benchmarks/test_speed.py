import time
import numpy as np
import unison_chat

rng = np.random.default_rng(1)
t0 = time.time()
r = unison_chat.reply("What is the capital of the fold?", rng)[0]
print("Gen:", r)
print("Time:", time.time() - t0)
