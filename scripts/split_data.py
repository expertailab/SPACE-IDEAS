import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_json("data/processed/space-ideas.jsonl",lines=True)
train, test = train_test_split(data, test_size=0.2, random_state=42)
                               
with open("data/processed/train.jsonl", "w") as f:
    f.write(train.to_json(orient="records",lines=True))

with open("data/processed/test.jsonl", "w") as f:
    f.write(test.to_json(orient="records",lines=True))
