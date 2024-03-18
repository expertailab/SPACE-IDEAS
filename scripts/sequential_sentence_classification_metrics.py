import argparse

from datasets import load_dataset

from ideas_annotation.modeling.idea_dataset_sentence_classification import (
    compute_predict_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Ideas sentence classification")
    parser.add_argument(
        "--prediction_test_file",
        type=str,
        help='Path to the predictions file.")',
    )
    parser.add_argument(
        "--gold_test_file",
        type=str,
        help="Path to the ground truth file",
    )
    args = parser.parse_args()

    return args


def main(args):

    predictions = load_dataset("json", data_files={"test": args.prediction_test_file})
    dataset = load_dataset("json", data_files={"test": args.gold_test_file})
    pred_labels = predictions["test"].map(
        lambda example: {
            "preds": [
                label.replace("_label", "") for sentence, label in example["preds"]
            ]
        }
    )["preds"]
    gold_labels = dataset["test"]["labels"]

    label2id = {"Challenge": 0, "Context": 1, "Solution": 2, "Benefits": 3, "Steps": 4}
    metrics = compute_predict_metrics(gold_labels, pred_labels, label2id)
    print(metrics)


if __name__ == "__main__":
    main(parse_args())
