import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from mixup import Mixup
from model.SFDINO import SFDINO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def scan_dataset(root):
    classes = sorted(
        entry.name for entry in os.scandir(root) if entry.is_dir()
    )
    if not classes:
        raise ValueError(f"No class directories found in: {root}")

    class_to_idx = {name: index for index, name in enumerate(classes)}
    samples = []
    for class_name in classes:
        class_dir = os.path.join(root, class_name)
        for current_root, dirnames, filenames in os.walk(class_dir):
            dirnames.sort()
            for filename in sorted(filenames):
                path = os.path.join(current_root, filename)
                if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                    samples.append((path, class_to_idx[class_name]))

    if not samples:
        raise ValueError(f"No supported images found in: {root}")
    return classes, samples


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_loaders(samples, train_indices, val_indices, test_indices, args):
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    def select(indices):
        return [samples[int(index)] for index in indices]

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        ImageDataset(select(train_indices), train_transform),
        shuffle=True,
        drop_last=False,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        ImageDataset(select(val_indices), eval_transform),
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        ImageDataset(select(test_indices), eval_transform),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, mixup_fn, device, epoch, epochs):
    model.train()
    loss_sum = 0.0
    sample_count = 0
    progress = tqdm(loader, desc=f"Train {epoch}/{epochs}", ncols=100)

    for images, labels in progress:
        if mixup_fn is not None and images.size(0) % 2 != 0:
            if images.size(0) == 1:
                continue
            images = images[:-1]
            labels = labels[:-1]

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if mixup_fn is None:
            mixed_images, mixed_labels = images, labels
            prototype_labels = labels
        else:
            mixed_images, mixed_labels = mixup_fn(images, labels)
            prototype_labels = mixed_labels.argmax(dim=1)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(mixed_images, labels=prototype_labels)
        loss = (
            criterion(outputs["logits"], mixed_labels)
            + criterion(outputs["logits_proto"], mixed_labels)
            + 0.5 * criterion(outputs["logits_linear"], mixed_labels)
        )
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{loss_sum / sample_count:.4f}")

    return loss_sum / sample_count


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images, labels=None)
        logits = outputs["logits"]
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        sample_count += batch_size
        for class_index in range(num_classes):
            mask = labels == class_index
            class_total[class_index] += mask.sum().item()
            class_correct[class_index] += ((predictions == labels) & mask).sum().item()

    class_accuracy = np.divide(
        class_correct,
        class_total,
        out=np.zeros(num_classes, dtype=np.float64),
        where=class_total != 0,
    )
    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
        "class_accuracy": class_accuracy.tolist(),
    }


def save_checkpoint(path, model, classes, fold, epoch, best_val_accuracy, split):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "num_classes": len(classes),
            "fold": fold,
            "epoch": epoch,
            "best_val_accuracy": best_val_accuracy,
            "split": split,
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Stratified five-fold training for SF-DINO")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", default="outputs/5fold")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", "--batchSize", type=int, default=24)
    parser.add_argument("--workers", "--threads", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no_mixup", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val_ratio must be between 0 and 1")
    if not args.no_mixup and args.batch_size % 2 != 0:
        raise ValueError(
            "--batch_size must be even when Mixup is enabled. "
            "Use an even batch size or pass --no_mixup."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    set_random_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes, samples = scan_dataset(args.dataset_root)
    targets = np.asarray([label for _, label in samples])
    paths = [os.path.abspath(path) for path, _ in samples]

    class_counts = np.bincount(targets, minlength=len(classes))
    if class_counts.min() < args.folds:
        raise ValueError(
            f"Each class needs at least {args.folds} images; counts={class_counts.tolist()}"
        )

    print(f"Device: {device}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Images: {len(samples)}, class counts: {class_counts.tolist()}")

    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    manifest = {
        "dataset_root": os.path.abspath(args.dataset_root),
        "classes": classes,
        "seed": args.seed,
        "folds": [],
    }
    fold_results = []

    for fold, (train_val_indices, test_indices) in enumerate(
        splitter.split(paths, targets), start=1
    ):
        fold_seed = args.seed + fold
        set_random_seed(fold_seed, args.deterministic)
        inner_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=args.val_ratio, random_state=fold_seed
        )
        inner_train, inner_val = next(
            inner_splitter.split(train_val_indices, targets[train_val_indices])
        )
        train_indices = train_val_indices[inner_train]
        val_indices = train_val_indices[inner_val]

        split = {
            "train_indices": train_indices.tolist(),
            "val_indices": val_indices.tolist(),
            "test_indices": test_indices.tolist(),
        }
        manifest["folds"].append({"fold": fold, **split})
        train_loader, val_loader, test_loader = make_loaders(
            samples, train_indices, val_indices, test_indices, args
        )

        model = SFDINO(num_classes=len(classes), pretrained=False).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        mixup_fn = None if args.no_mixup else Mixup(
            mixup_alpha=0.4,
            cutmix_alpha=0.0,
            switch_prob=0.0,
            mode="batch",
            label_smoothing=0.1,
            num_classes=len(classes),
        )
        checkpoint_path = os.path.join(args.output_dir, f"fold_{fold}_best.pth")
        history_path = os.path.join(args.output_dir, f"fold_{fold}_history.csv")
        best_val_accuracy = -1.0
        best_epoch = 0

        print(
            f"\nFold {fold}/{args.folds}: train={len(train_indices)}, "
            f"val={len(val_indices)}, test={len(test_indices)}"
        )
        with open(history_path, "w", newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(
                history_file,
                fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"],
            )
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                train_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, mixup_fn,
                    device, epoch, args.epochs
                )
                val_metrics = evaluate(
                    model, val_loader, criterion, device, len(classes)
                )
                writer.writerow({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                })
                history_file.flush()
                print(
                    f"Fold {fold} epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_acc={val_metrics['accuracy']:.4f}"
                )
                if val_metrics["accuracy"] > best_val_accuracy:
                    best_val_accuracy = val_metrics["accuracy"]
                    best_epoch = epoch
                    save_checkpoint(
                        checkpoint_path, model, classes, fold, epoch,
                        best_val_accuracy, split
                    )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(
            model, test_loader, criterion, device, len(classes)
        )
        fold_result = {
            "fold": fold,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "test_accuracy": test_metrics["accuracy"],
            "test_class_accuracy": test_metrics["class_accuracy"],
        }
        fold_results.append(fold_result)
        print(f"Fold {fold} test accuracy: {test_metrics['accuracy']:.4f}")

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracies = np.asarray([item["test_accuracy"] for item in fold_results])
    summary = {
        "classes": classes,
        "fold_results": fold_results,
        "mean_test_accuracy": float(accuracies.mean()),
        "std_test_accuracy": float(accuracies.std(ddof=1)),
    }
    with open(
        os.path.join(args.output_dir, "splits.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    with open(
        os.path.join(args.output_dir, "train_summary.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(
        f"\nFive-fold accuracy: {summary['mean_test_accuracy']:.4f} "
        f"+/- {summary['std_test_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
