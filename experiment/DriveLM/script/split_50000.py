import json
import random

src="data/train_llama.json"
dst="data/train_llama_50000.json"

with open(src) as f:
    data=json.load(f)

random.seed(42)

small=random.sample(data,50000)

with open(dst,"w") as f:
    json.dump(small,f,indent=4)

print("original:",len(data))
print("sample:",len(small))
