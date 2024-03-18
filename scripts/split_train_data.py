import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_json("data/processed/train.jsonl",lines=True)
train, dev = train_test_split(data, test_size=0.2, random_state=42)
                               
with open("data/processed/train2.jsonl", "w") as f:
    f.write(train.to_json(orient="records",lines=True))

with open("data/processed/dev.jsonl", "w") as f:
    f.write(dev.to_json(orient="records",lines=True))
