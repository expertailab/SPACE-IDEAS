import copy
import itertools

from typing import List
from datetime import datetime

import numpy as np

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from grouphug import (
    DatasetFormatter,
    MultiTaskTrainer,
    AutoMultiTaskModel,
    ClassificationHead,
    SequenceClassificationHeadConfig,
)
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_preds, dataset_name, heads):
    logits, labels = eval_preds
    mask = labels != -100
    labels = labels[mask].flatten()
    predictions = np.argmax(logits[mask], axis=-1)
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


tokenizer = AutoTokenizer.from_pretrained("roberta-large")
predict = False


def text_to_instance(sentences: List[str], labels: List[str] = None):
    if not predict:
        assert len(sentences) == len(labels)

    origin_sent = copy.deepcopy(sentences)
    sentences = shorten_sentences(sentences, 112)

    max_len = 112
    while len(sentences) > 512:
        n = int((len(sentences) - 512) / len(origin_sent)) + 1

        max_len -= n
        sentences = shorten_sentences(origin_sent, max_len)

    assert len(sentences) <= 512

    fields = {}
    fields["input_ids"] = sentences

    if labels is not None:
        fields["labels"] = labels

    return fields


def shorten_sentences(origin_sent, max_len):
    tokenized_sentences = [[tokenizer.cls_token_id]]
    for s in origin_sent:
        if len(tokenizer(s, add_special_tokens=True)["input_ids"]) > (max_len):
            tokenized_sentences.append(
                tokenizer(s, add_special_tokens=True)["input_ids"][1:(max_len)]
                + [tokenizer.eos_token_id, tokenizer.sep_token_id]
            )
        else:
            tokenized_sentences.append(
                tokenizer(s, add_special_tokens=True)["input_ids"][1:-1]
                + [tokenizer.eos_token_id, tokenizer.sep_token_id]
            )
    mid_tok_len = len([tokenizer.eos_token_id, tokenizer.sep_token_id])
    return list(itertools.chain.from_iterable(tokenized_sentences))[:-mid_tok_len] + [
        tokenizer.eos_token_id
    ]


max_sent_per_example = 10


def enforce_max_sent_per_example(sentences, labels=None):
    """
    Splits examples with len(sentences) > self.max_sent_per_example into multiple
    smaller examples     with len(sentences) <= self.max_sent_per_example.
    Recursively split the list of sentences into two halves until each half
    has len(sentences) < <= self.max_sent_per_example. The goal is to produce splits
    that are of almost equal size to avoid the scenario where all splits are of size
    self.max_sent_per_example then the last split is 1 or 2 sentences
    This will result into losing context around the edges of each examples.
    """
    if labels is not None:
        assert len(sentences) == len(labels)

    if len(sentences) > max_sent_per_example and max_sent_per_example > 0:
        i = len(sentences) // 2
        l1 = enforce_max_sent_per_example(
            sentences[:i], None if labels is None else labels[:i]
        )
        l2 = enforce_max_sent_per_example(
            sentences[i:], None if labels is None else labels[i:]
        )
        return l1 + l2
    else:
        return [(sentences, labels)]


def read_one_example(json_dict):
    instances = []
    sentences = json_dict["sentences"]

    if not predict:
        labels = json_dict["labels"]
    else:
        labels = None

    for sentences_loop, labels_loop in enforce_max_sent_per_example(sentences, labels):

        instance = text_to_instance(sentences=sentences_loop, labels=labels_loop)
        instances.append(instance)
    return instances


def main():

    tasks = ["gold", "csabstruct"]
    # tasks = ["csabstruct", "pubmed", "chatgpt", "gold"]

    label2id = {"Challenge": 0, "Context": 1, "Solution": 2, "Benefits": 3, "Steps": 4}
    dataset = load_from_disk("data/processed/merge_ideas_dataset")
    gold_dataset = dataset["train"].filter(
        lambda example: not example["description"] and len(example["labels"]) > 0
    )
    gold_dataset = gold_dataset.map(
        lambda example: {
            "sentences": [
                sentence
                for sentence, label in zip(
                    example["sentences"][1:], example["labels"][1:]
                )
                if label != "None"
            ],
            "labels": [
                label2id[label] for label in example["labels"][1:] if label != "None"
            ],
        }
    )
    gold_dataset = Dataset.from_list(
        [instance for example in gold_dataset for instance in read_one_example(example)]
    ).rename_column("labels", "gold")
    validation_dataset = dataset["validation"].map(
        lambda example: {
            "sentences": example["sentences"][1:],
            "labels": [label2id[label] for label in example["labels"][1:]],
        }
    )
    validation_dataset = Dataset.from_list(
        [
            instance
            for example in validation_dataset
            for instance in read_one_example(example)
        ]
    )
    gold_dataset = DatasetDict(
        {
            "train": gold_dataset,
            "validation": validation_dataset.rename_column("labels", "gold"),
        }
    )
    data = {"gold": gold_dataset}

    if "chatgpt" in tasks:
        chatgpt_dataset = dataset["train"].filter(
            lambda example: example["description"] and len(example["labels"]) > 0
        )
        chatgpt_dataset = chatgpt_dataset.map(
            lambda example: {"labels": [label2id[label] for label in example["labels"]]}
        )
        chatgpt_dataset = Dataset.from_list(
            [
                instance
                for example in chatgpt_dataset
                for instance in read_one_example(example)
            ]
        ).rename_column("labels", "chatgpt")
        chatgpt_dataset = DatasetDict(
            {
                "train": chatgpt_dataset,
                "validation": validation_dataset.rename_column("labels", "chatgpt"),
            }
        )
        data["chatgpt"] = chatgpt_dataset

    if "csabstruct" in tasks:
        csabstruct_dataset = load_dataset("allenai/csabstruct")
        csabstruct_train = Dataset.from_list(
            [
                instance
                for example in csabstruct_dataset["train"]
                for instance in read_one_example(example)
            ]
        ).rename_column("labels", "csabstruct")
        csabstruct_validation = Dataset.from_list(
            [
                instance
                for example in csabstruct_dataset["validation"]
                for instance in read_one_example(example)
            ]
        ).rename_column("labels", "csabstruct")
        csabstruct_dataset = DatasetDict(
            {"train": csabstruct_train, "validation": csabstruct_validation}
        )
        data["csabstruct"] = csabstruct_dataset

    if "pubmed" in tasks:
        labels = ["BACKGROUND", "METHODS", "OBJECTIVE", "CONCLUSIONS", "RESULTS"]
        label2id = {k: i for i, k in enumerate(labels)}
        pubmed_dataset = load_from_disk("data/processed/pubmed-20k-rct")
        pubmed_dataset = pubmed_dataset.map(
            lambda example: {"labels": [label2id[label] for label in example["labels"]]}
        )
        pubmed_train = Dataset.from_list(
            [
                instance
                for example in pubmed_dataset["train"]
                for instance in read_one_example(example)
            ]
        ).rename_column("labels", "pubmed")
        pubmed_validation = Dataset.from_list(
            [
                instance
                for example in pubmed_dataset["dev"]
                for instance in read_one_example(example)
            ]
        ).rename_column("labels", "pubmed")
        pubmed_dataset = DatasetDict(
            {"train": pubmed_train, "validation": pubmed_validation}
        )
        data["pubmed"] = pubmed_dataset

    fmt = DatasetFormatter()

    tokenized_data = fmt.apply(
        data, tokenizer=tokenizer, splits=["train", "validation"]
    )

    head_configs = [
        SequenceClassificationHeadConfig.from_data(
            tokenized_data,
            task,
            detached=False,
            ignore_index=-1,
            problem_type=ClassificationHead.SINGLE,
        )
        for task in tasks
    ]

    base_model = "roberta-large"
    model = AutoMultiTaskModel.from_pretrained(
        base_model, head_configs, formatter=fmt, tokenizer=tokenizer
    )

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y%m%dT%H%M%S")
    output_dir = (
        "multitask_sequential_sentence_classification_osip_"
        + "_".join(tasks)
        + "-"
        + date_time
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=1,
        num_train_epochs=20,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_gold_f1",
        warmup_ratio=0.1,
        logging_steps=10,
        report_to="wandb",
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

    trainer.train()


if __name__ == "__main__":
    main()
