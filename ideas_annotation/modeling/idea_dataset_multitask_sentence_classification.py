from datetime import datetime

import numpy as np

from datasets import DatasetDict, load_dataset, load_from_disk
from grouphug import (
    DatasetFormatter,
    MultiTaskTrainer,
    AutoMultiTaskModel,
    ClassificationHeadConfig,
)
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ideas_annotation.modeling.scim_sentence_classification import scim_to_sentences
from ideas_annotation.modeling.pubmed_sentence_classification import pubmed_to_sentences
from ideas_annotation.modeling.csabstruct_sentence_classification import (
    csabstruct_to_sentences,
)
from ideas_annotation.modeling.idea_dataset_sentence_classification import (
    ideas_to_sentences,
)

now = datetime.now()  # current date and time
date_time = now.strftime("%Y%m%dT%H%M%S")


def compute_metrics(eval_preds, dataset_name, heads):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    acc = accuracy_score(labels, predictions)
    #     conff = confusion_matrix(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
        #         'conff' : conff
    }


tasks = ["scim", "csabstruct", "pubmed", "chatgpt", "gold"]

label2id = {"Challenge": 0, "Context": 1, "Solution": 2, "Benefits": 3, "Steps": 4}
ideas_dataset = load_from_disk("data/processed/merge_ideas_dataset")
gold_dataset = ideas_dataset["train"].filter(lambda example: not example["description"])
gold_dataset_sentences = ideas_to_sentences(gold_dataset).rename_column("label", "gold")
gold_dataset_sentences = gold_dataset_sentences.filter(
    lambda example: example["gold"] != "None"
)
gold_dataset_sentences = gold_dataset_sentences.map(
    lambda batch: {"gold": label2id[batch["gold"]]}
)
validation_sentences = ideas_to_sentences(ideas_dataset["validation"])
validation_sentences = validation_sentences.map(
    lambda batch: {"label": label2id[batch["label"]]}
)
gold_dataset = DatasetDict(
    {
        "train": gold_dataset_sentences,
        "validation": validation_sentences.rename_column("label", "gold"),
    }
)
data = {"gold": gold_dataset}

if "chatgpt" in tasks:
    chatgpt_dataset = ideas_dataset["train"].filter(
        lambda example: example["description"]
    )
    chatgpt_dataset_sentences = (
        ideas_to_sentences(chatgpt_dataset)
        .rename_column("label", "chatgpt")
        .map(lambda batch: {"chatgpt": label2id[batch["chatgpt"]]})
    )
    chatgpt_dataset = DatasetDict(
        {
            "train": chatgpt_dataset_sentences,
            "validation": validation_sentences.rename_column("label", "chatgpt"),
        }
    )
    data["chatgpt"] = chatgpt_dataset

if "csabstruct" in tasks:
    csabstruct_dataset = load_dataset("allenai/csabstruct")
    csabstruct_train_dataset = csabstruct_to_sentences(csabstruct_dataset["train"])
    csabstruct_validation_dataset = csabstruct_to_sentences(
        csabstruct_dataset["validation"]
    )
    csabstruct_dataset = DatasetDict(
        {
            "train": csabstruct_train_dataset.rename_column("label", "csabstruct"),
            "validation": csabstruct_validation_dataset.rename_column(
                "label", "csabstruct"
            ),
        }
    )
    data["csabstruct"] = csabstruct_dataset

if "pubmed" in tasks:
    labels = ["BACKGROUND", "METHODS", "OBJECTIVE", "CONCLUSIONS", "RESULTS"]
    label2id = {k: i for i, k in enumerate(labels)}
    pubmed_dataset = load_from_disk("data/processed/pubmed-20k-rct")
    # Random sample of 3k instances
    pubmed_dataset["train"] = pubmed_dataset["train"].shuffle(seed=42)
    pubmed_dataset["train"] = pubmed_dataset["train"].select(range(3000))
    pubmed_train_dataset = pubmed_to_sentences(pubmed_dataset["train"])
    pubmed_train_dataset = pubmed_train_dataset.map(
        lambda batch: {"label": label2id[batch["label"]]}
    )
    pubmed_validation_dataset = pubmed_to_sentences(pubmed_dataset["dev"])
    pubmed_validation_dataset = pubmed_validation_dataset.map(
        lambda batch: {"label": label2id[batch["label"]]}
    )
    pubmed_dataset = DatasetDict(
        {
            "train": pubmed_train_dataset.rename_column("label", "pubmed"),
            "validation": pubmed_validation_dataset.rename_column("label", "pubmed"),
        }
    )
    data["pubmed"] = pubmed_dataset

if "scim" in tasks:
    labels = ["abstain", "method", "objective", "other", "result"]
    label2id = {k: i for i, k in enumerate(labels)}
    id2label = {i: k for i, k in enumerate(labels)}

    scim_dataset = load_from_disk("data/processed/scim")
    scim_train_dataset = scim_to_sentences(scim_dataset["train"])
    scim_train_dataset = scim_train_dataset.map(
        lambda batch: {"label": label2id[batch["label"]]}
    )
    scim_validation_dataset = scim_to_sentences(scim_dataset["validation"])
    scim_validation_dataset = scim_validation_dataset.map(
        lambda batch: {"label": label2id[batch["label"]]}
    )
    scim_dataset = DatasetDict(
        {
            "train": scim_train_dataset.rename_column("label", "scim"),
            "validation": scim_validation_dataset.rename_column("label", "scim"),
        }
    )
    data["scim"] = scim_dataset


base_model = "roberta-large"

tokenizer = AutoTokenizer.from_pretrained(base_model)

fmt = (
    DatasetFormatter()
    .tokenize(max_length=512)
    .tokenize(("sentence", "context"), max_length=512)
)

tokenized_data = fmt.apply(data, tokenizer=tokenizer, splits=["train", "validation"])

head_configs = [
    ClassificationHeadConfig.from_data(
        tokenized_data, task, detached=False, ignore_index=-1
    )
    for task in tasks
]

model = AutoMultiTaskModel.from_pretrained(
    base_model, head_configs, formatter=fmt, tokenizer=tokenizer
)

output_dir = (
    "multitask_training_output_scim_csabstruct_pubmed-3k_osip+_osip" + "-" + date_time
)
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="eval_gold_f1",
    logging_steps=10,
)

trainer = MultiTaskTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_data=tokenized_data[:, "train"],
    eval_data=tokenized_data[:, "validation"],
    eval_heads={t: [t] for t in tasks},  # for dataset [key], run heads [value]
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

train_res = trainer.train()
