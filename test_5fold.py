import argparse
import json
import os

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model.SFDINO import SFDINO
from train_5fold import ImageDataset, scan_dataset


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels_all = []
    predictions_all = []
    probabilities_all = []
    for images, labels in tqdm(loader, desc="Testing", ncols=100):
        images = images.to(device, non_blocking=True)
        logits = model(images, labels=None)["logits"]
        probabilities = F.softmax(logits, dim=1)
        labels_all.extend(labels.numpy().tolist())
        predictions_all.extend(probabilities.argmax(dim=1).cpu().numpy().tolist())
        probabilities_all.extend(probabilities.cpu().numpy().tolist())
    return (
        np.asarray(labels_all),
        np.asarray(predictions_all),
        np.asarray(probabilities_all),
    )


def save_confusion_matrix(matrix, classes, path):
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="Five-fold out-of-fold confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Test SF-DINO five-fold checkpoints")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--checkpoint_dir", default="outputs/5fold")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", "--batchSize", type=int, default=24)
    parser.add_argument("--workers", "--threads", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes, samples = scan_dataset(args.dataset_root)
    manifest_path = os.path.join(args.checkpoint_dir, "splits.json")
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if classes != manifest["classes"]:
        raise ValueError(
            f"Dataset classes {classes} do not match training classes "
            f"{manifest['classes']}"
        )

    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    all_labels = []
    all_predictions = []
    all_probabilities = []
    fold_results = []

    for fold_info in manifest["folds"]:
        fold = fold_info["fold"]
        test_indices = fold_info["test_indices"]
        test_samples = [samples[int(index)] for index in test_indices]
        loader = DataLoader(
            ImageDataset(test_samples, eval_transform),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        checkpoint_path = os.path.join(
            args.checkpoint_dir, f"fold_{fold}_best.pth"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint["classes"] != classes:
            raise ValueError(f"Class mismatch in checkpoint: {checkpoint_path}")

        model = SFDINO(num_classes=len(classes), pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        labels, predictions, probabilities = predict(model, loader, device)
        accuracy = accuracy_score(labels, predictions)
        fold_results.append({
            "fold": fold,
            "checkpoint": checkpoint_path,
            "best_epoch": checkpoint["epoch"],
            "accuracy": float(accuracy),
            "macro_f1": float(f1_score(
                labels, predictions, average="macro", zero_division=0
            )),
        })
        all_labels.append(labels)
        all_predictions.append(predictions)
        all_probabilities.append(probabilities)
        print(f"Fold {fold}: accuracy={accuracy:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    matrix = confusion_matrix(
        labels, predictions, labels=np.arange(len(classes))
    )
    fold_accuracies = np.asarray([item["accuracy"] for item in fold_results])
    report = classification_report(
        labels,
        predictions,
        labels=np.arange(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "fold_results": fold_results,
        "mean_fold_accuracy": float(fold_accuracies.mean()),
        "std_fold_accuracy": float(fold_accuracies.std(ddof=1)),
        "oof_accuracy": float(accuracy_score(labels, predictions)),
        "oof_macro_precision": float(precision_score(
            labels, predictions, average="macro", zero_division=0
        )),
        "oof_macro_recall": float(recall_score(
            labels, predictions, average="macro", zero_division=0
        )),
        "oof_macro_f1": float(f1_score(
            labels, predictions, average="macro", zero_division=0
        )),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    with open(
        os.path.join(output_dir, "test_summary.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    np.savez(
        os.path.join(output_dir, "oof_predictions.npz"),
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
    )
    save_confusion_matrix(
        matrix, classes, os.path.join(output_dir, "confusion_matrix.png")
    )

    print(
        f"\nMean fold accuracy: {summary['mean_fold_accuracy']:.4f} "
        f"+/- {summary['std_fold_accuracy']:.4f}"
    )
    print(
        f"OOF accuracy={summary['oof_accuracy']:.4f}, "
        f"macro F1={summary['oof_macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
