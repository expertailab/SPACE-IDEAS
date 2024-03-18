from datasets import load_dataset, DatasetDict, concatenate_datasets
gold_dataset = load_dataset("json", data_files ={"train":"data/processed/train.jsonl","test":"data/processed/test.jsonl"})
gold_dataset_split = gold_dataset["train"].train_test_split(0.2)
dataset = DatasetDict({"train":gold_dataset_split["train"], "validation":gold_dataset_split["test"], "test": gold_dataset["test"]})
chatgpt_dataset = load_dataset("json", data_files ="data/processed/space-ideas_plus.jsonl")
chatgpt_dataset_filtered = chatgpt_dataset.filter(lambda example: example["doc_id"] not in gold_dataset["train"]["doc_id"]+gold_dataset["test"]["doc_id"])
concatenated_train = concatenate_datasets([dataset["train"],chatgpt_dataset_filtered["train"]])
dataset = DatasetDict({"train":concatenated_train, "validation":dataset["validation"], "test": dataset["test"]})
dataset.save_to_disk ("data/processed/merge_ideas_dataset")
