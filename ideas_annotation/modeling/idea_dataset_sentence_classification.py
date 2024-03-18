import argparse
import itertools

import numpy as np
import torch

from datasets import Dataset, load_dataset
from transformers import (
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans


def sequence_to_bio(sequence):
    output = []
    for i, element in enumerate(sequence):
        if i == 0 or sequence[i - 1] != element:
            output.append("B-" + element)
        else:
            output.append("I-" + element)
    return output


def span_f1(gold_labels, predict_labels):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for gold_labels_element, predict_labels_element in zip(gold_labels, predict_labels):
        gold_spans = bio_tags_to_spans(sequence_to_bio(gold_labels_element))
        predict_spans = bio_tags_to_spans(sequence_to_bio(predict_labels_element))
        for span in predict_spans:
            if span in gold_spans:
                true_positives += 1
                gold_spans.remove(span)
            else:
                false_positives += 1
        for span in gold_spans:
            false_negatives += 1
    precision = true_positives / (true_positives + false_positives + 1e-13)
    recall = true_positives / (true_positives + false_negatives + 1e-13)
    f1_measure = 2.0 * (precision * recall) / (precision + recall + 1e-13)
    return f1_measure


def preprocess_test_set_function(examples, tokenizer=None, use_context=False):
    if use_context:
        return tokenizer(
            examples["sentences"],
            text_pair=[examples["context"]] * len(examples["sentences"]),
            truncation=True,
            return_tensors="pt",
            padding=True,
        )
    else:
        return tokenizer(
            examples["sentences"], truncation=True, return_tensors="pt", padding=True
        )


def preprocess_function(examples, tokenizer=None, use_context=False):
    if use_context:
        return tokenizer(
            examples["sentence"], text_pair=examples["context"], truncation=True
        )
    else:
        return tokenizer(examples["sentence"], truncation=True)


def predict(examples, model=None):
    model.eval()
    with torch.no_grad():
        logits = model(**{key: value.to(0) for key, value in examples.items()}).logits
    pred_labels = [model.config.id2label[id_] for id_ in logits.argmax(-1).tolist()]
    return {"pred_labels": pred_labels}


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    #     conff = confusion_matrix(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
        #         'conff' : conff
    }


def compute_predict_metrics(gold_labels, predict_labels, label2id):
    micro_span_f1 = span_f1(gold_labels, predict_labels)
    labels = list(itertools.chain(*gold_labels))
    preds = list(itertools.chain(*predict_labels))
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro"
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    metrics_label = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(label2id.keys())
    )
    conf_matrix = confusion_matrix(labels, preds, labels=list(label2id.keys()))
    #     conff = confusion_matrix(labels, preds)
    return {
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_span_f1": micro_span_f1,
        "metrics_label": metrics_label,
        "confusion_matrix": conf_matrix,
    }


def ideas_to_sentences(dataset):
    sentences = []
    labels = []
    contexts = []
    for example in dataset:
        sentences += example["sentences"]
        labels += example["labels"]
        title = example["title"] if example["title"] else ""
        description = example["description"] if ("description" in example and example["description"]) else " ".join(example["sentences"])
        context = title + "\n" + description
        contexts += [context] * len(example["labels"])
    ds = Dataset.from_dict(
        {"sentence": sentences, "label": labels, "context": contexts}
    )
    return ds


def parse_args():
    parser = argparse.ArgumentParser(description="Ideas sentence classification")
    parser.add_argument(
        "--input_train_dataset",
        type=str,
        default="data/processed/train.jsonl",
        help='Path to the training dataset (default "data/processed/train.jsonl")',
    )
    parser.add_argument(
        "--input_test_dataset",
        type=str,
        default="data/processed/test.jsonl",
        help='Path to the test dataset (default "data/processed/test.jsonl")',
    )
    parser.add_argument(
        "--use_context",
        default=False,
        action="store_true",
        help="Whether to use context (the whole idea) for the sentence classification",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="roberta-large",
        help='Huggingface model to finetune (default "roberta-large")',
    )
    args = parser.parse_args()

    return args


def main(args):
    id2label = {0: "Challenge", 1: "Context", 2: "Solution", 3: "Benefits", 4: "Steps"}
    label2id = {"Challenge": 0, "Context": 1, "Solution": 2, "Benefits": 3, "Steps": 4}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.input_train_dataset,
        },
    )
    train_dataset = ideas_to_sentences(dataset["train"])
    train_dataset = train_dataset.filter(lambda example: example["label"] != "None")
    train_dataset = train_dataset.map(lambda batch: {"label": label2id[batch["label"]]})
    train_tokenized_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "use_context": args.use_context},
    )
    dataset = load_dataset(
        "json",
        data_files={
            "test": args.input_test_dataset,
        },
    )
    test_dataset = dataset["test"].map(
        lambda example: {
            "sentences": example["sentences"][1:],
            "labels": example["labels"][1:],
            **(
                {
                    "context": example["sentences"][0]
                    + "\n"
                    + " ".join(example["sentences"][1:])
                }
                if args.use_context
                else {}
            ),
        }
    )
    test_tokenized_dataset = test_dataset.map(
        preprocess_test_set_function,
        fn_kwargs={"tokenizer": tokenizer, "use_context": args.use_context},
    )

    training_args = TrainingArguments(
        output_dir="my_awesome_model",
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
        metric_for_best_model="eval_f1",
        logging_steps=10,
    )

    train2 = train_tokenized_dataset.train_test_split(test_size=0.2)

    num_runs = 3
    all_metrics = []
    for i in range(num_runs):
        print(f"Starting training run {i + 1}")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=5, id2label=id2label, label2id=label2id
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train2["train"],
            eval_dataset=train2["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.train()
        pred = (
            test_tokenized_dataset.remove_columns(
                ["doc_id", "sentences", "labels"]
                + (["context" if args.use_context else []])
            )
            .with_format("torch")
            .map(predict, fn_kwargs={"model": model})
        )
        pred_labels = pred["pred_labels"]
        gold_labels = test_tokenized_dataset["labels"]
        metrics = compute_predict_metrics(gold_labels, pred_labels, label2id)
        print(metrics)
        all_metrics.append(metrics)

    print(all_metrics)
    mean_micro_f1_score = np.array(
        [metric["micro_f1"] for metric in all_metrics]
    ).mean()
    mean_macro_f1_score = np.array(
        [metric["macro_f1"] for metric in all_metrics]
    ).mean()
    mean_micro_span_f1_score = np.array(
        [metric["micro_span_f1"] for metric in all_metrics]
    ).mean()

    print("mean_micro_f1_score:", mean_micro_f1_score)
    print("mean_macro_f1_score:", mean_macro_f1_score)
    print("mean_micro_span_f1_score:", mean_micro_span_f1_score)


if __name__ == "__main__":
    main(parse_args())
