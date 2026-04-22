from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from midi2tag_model import (
    MIDI2TAG_FIELDS,
    Midi2TagModel,
    build_remi_tokenizer,
    midi2tag_label_maps,
    save_midi2tag_model,
    tokenizer_vocab_size,
)


class Midi2TagDataset(Dataset):
    def __init__(self, path: str | Path, max_seq_len: int) -> None:
        self.rows = json.loads(Path(path).read_text(encoding="utf-8"))
        self.max_seq_len = max_seq_len
        self.forward_maps, _ = midi2tag_label_maps(MIDI2TAG_FIELDS)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        tokens = [int(token_id) for token_id in row["tokens"][: self.max_seq_len]]
        labels = {
            field: self.forward_maps[field][row["labels"][field]]
            for field in MIDI2TAG_FIELDS
        }
        return {
            "tokens": tokens,
            "labels": labels,
            "midi_path": row.get("midi_path"),
            "caption": row.get("caption", ""),
        }


def collate_fn(batch: list[dict[str, Any]], pad_id: int = 0) -> dict[str, Any]:
    max_len = max(len(item["tokens"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for row_index, item in enumerate(batch):
        tokens = torch.tensor(item["tokens"], dtype=torch.long)
        input_ids[row_index, : len(tokens)] = tokens
        attention_mask[row_index, : len(tokens)] = 1

    labels = {
        field: torch.tensor([item["labels"][field] for item in batch], dtype=torch.long)
        for field in MIDI2TAG_FIELDS
    }
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def evaluate(model: Midi2TagModel, dataloader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    totals = {field: 0 for field in MIDI2TAG_FIELDS}
    correct = {field: 0 for field in MIDI2TAG_FIELDS}
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            labels = {field: tensor.to(device) for field, tensor in batch.pop("labels").items()}
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = 0.0
            for field, field_logits in logits.items():
                loss += F.cross_entropy(field_logits, labels[field])
                predictions = torch.argmax(field_logits, dim=-1)
                correct[field] += int((predictions == labels[field]).sum().item())
                totals[field] += int(labels[field].numel())
            total_loss += float((loss / len(MIDI2TAG_FIELDS)).item())
            batch_count += 1

    metrics = {"loss": total_loss / max(batch_count, 1)}
    for field in MIDI2TAG_FIELDS:
        metrics[f"{field}_accuracy"] = correct[field] / max(totals[field], 1)
    metrics["mean_accuracy"] = sum(metrics[f"{field}_accuracy"] for field in MIDI2TAG_FIELDS) / len(MIDI2TAG_FIELDS)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight REMI-token MIDI2Tags classifier.")
    parser.add_argument("--train", default="data/midi2tag/train.json")
    parser.add_argument("--val", default="data/midi2tag/val.json")
    parser.add_argument("--output-dir", default="artifacts/midi2tag")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = build_remi_tokenizer()
    vocab_size = tokenizer_vocab_size(tokenizer)
    pad_id = int(tokenizer.vocab["PAD_None"])

    train_dataset = Midi2TagDataset(args.train, max_seq_len=args.max_seq_len)
    val_dataset = Midi2TagDataset(args.val, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id),
    )

    model = Midi2TagModel(
        vocab_size=vocab_size,
        fields=MIDI2TAG_FIELDS,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pad_id=pad_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    train_losses: list[float] = []
    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            labels = {field: tensor.to(device) for field, tensor in batch.pop("labels").items()}
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = sum(F.cross_entropy(field_logits, labels[field]) for field, field_logits in logits.items())
            loss = loss / len(MIDI2TAG_FIELDS)

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

    metadata = {
        "fields": MIDI2TAG_FIELDS,
        "vocab_size": vocab_size,
        "pad_id": pad_id,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "max_seq_len": args.max_seq_len,
        "train_args": vars(args),
        "metrics": metrics,
    }
    save_midi2tag_model(args.output_dir, model, metadata)
    output_dir = Path(args.output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
