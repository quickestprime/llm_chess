import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel, AdamW
import chess
import torch.nn as nn
import argparse
import os

class ChessMoveDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=128, max_samples=None):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r") as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                sample = json.loads(line)
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fen = sample["fen"]
        piece_type = sample["piece_type"]  # 0-5
        destination_square = sample["destination_square"]  # 0-63

        encoding = self.tokenizer(
            fen,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Compute legal moves mask for destination squares
        board = chess.Board(fen)
        legal_mask = torch.zeros(64, dtype=torch.bool)
        for move in board.legal_moves:
            legal_mask[move.to_square] = True

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "piece_type": torch.tensor(piece_type, dtype=torch.long),
            "destination_square": torch.tensor(destination_square, dtype=torch.long),
            "legal_mask": legal_mask
        }

class ChessMoveModel(nn.Module):
    def __init__(self, pretrained_model="distilbert-base-uncased"):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model)
        hidden_size = self.bert.config.hidden_size

        # Separate classifiers for piece type and destination square
        self.piece_classifier = nn.Linear(hidden_size, 6)   # 6 piece classes
        self.dest_classifier = nn.Linear(hidden_size, 64)   # 64 squares

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]  # CLS token output

        piece_logits = self.piece_classifier(hidden_state)
        dest_logits = self.dest_classifier(hidden_state)

        return piece_logits, dest_logits

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataset = ChessMoveDataset(args.data_path, tokenizer, max_length=128, max_samples=args.max_samples)
    print(f"Loaded dataset with {len(dataset)} samples")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ChessMoveModel().to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            piece_labels = batch["piece_type"].to(device)
            dest_labels = batch["destination_square"].to(device)
            legal_mask = batch["legal_mask"].to(device)

            optimizer.zero_grad()
            piece_logits, dest_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Mask illegal destination squares before loss
            mask_value = -1e9
            dest_logits_masked = dest_logits.masked_fill(~legal_mask, mask_value)

            loss_piece = loss_fn(piece_logits, piece_labels)
            loss_dest = loss_fn(dest_logits_masked, dest_labels)

            loss = loss_piece + loss_dest
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="magnus_piece_dest_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="./bert_chess_piece_dest")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    train(args)
