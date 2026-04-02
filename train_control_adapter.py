from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from control_adapter import CONTROL_FIELDS, ControlAdapterModel, label_maps, save_adapter


class JsonDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.rows = json.loads(Path(path).read_text(encoding="utf-8"))
        self.forward_maps, _ = label_maps()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        labels = {field: self.forward_maps[field][row[field]] for field in CONTROL_FIELDS}
        return {"caption": row["caption"], "labels": labels}


def collate_fn(batch: list[dict], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    captions = [item["caption"] for item in batch]
    encoded = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    label_tensors = {
        field: torch.tensor([item["labels"][field] for item in batch], dtype=torch.long) for field in CONTROL_FIELDS
    }
    encoded["labels"] = label_tensors
    return encoded


def evaluate(model: ControlAdapterModel, dataloader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    totals = {field: 0 for field in CONTROL_FIELDS}
    correct = {field: 0 for field in CONTROL_FIELDS}
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in dataloader:
            labels = {field: tensor.to(device) for field, tensor in batch.pop("labels").items()}
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch)
            batch_loss = 0.0
            for field, field_logits in logits.items():
                field_loss = F.cross_entropy(field_logits, labels[field])
                batch_loss += field_loss
                preds = torch.argmax(field_logits, dim=-1)
                correct[field] += int((preds == labels[field]).sum().item())
                totals[field] += int(labels[field].numel())
            total_loss += float(batch_loss.item() / len(CONTROL_FIELDS))
            batch_count += 1

    metrics = {"loss": total_loss / max(batch_count, 1)}
    for field in CONTROL_FIELDS:
        metrics[f"{field}_accuracy"] = correct[field] / max(totals[field], 1)
    metrics["mean_accuracy"] = sum(metrics[f"{field}_accuracy"] for field in CONTROL_FIELDS) / len(CONTROL_FIELDS)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multi-head caption-to-controls adapter.")
    parser.add_argument("--train", default="data/control_adapter/train.json")
    parser.add_argument("--val", default="data/control_adapter/val.json")
    parser.add_argument("--output-dir", default="artifacts/control_adapter")
    parser.add_argument("--encoder-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-train-steps", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    train_dataset = JsonDataset(args.train)
    val_dataset = JsonDataset(args.val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    model = ControlAdapterModel(encoder_name=args.encoder_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    train_losses: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            labels = {field: tensor.to(device) for field, tensor in batch.pop("labels").items()}
            batch = {key: value.to(device) for key, value in batch.items()}

            logits = model(**batch)
            loss = sum(F.cross_entropy(field_logits, labels[field]) for field, field_logits in logits.items())
            loss = loss / len(CONTROL_FIELDS)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            global_step += 1
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    metrics = evaluate(model, val_loader, device)
    metrics["train_loss_last"] = train_losses[-1] if train_losses else None
    metrics["train_steps"] = global_step

    save_adapter(
        args.output_dir,
        model,
        encoder_name=args.encoder_name,
        metadata={
            "metrics": metrics,
            "train_args": vars(args),
        },
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
