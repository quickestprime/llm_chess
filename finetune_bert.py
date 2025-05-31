import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
import chess
import argparse
import os

# --- Dataset class ---

class ChessMoveDataset(Dataset):
    def __init__(self, jsonl_path, move_vocab_path, tokenizer, max_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load samples from jsonl
        with open(jsonl_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)

        # Load full move vocab
        with open(move_vocab_path, "r") as f:
            self.move_vocab = json.load(f)
        self.idx_to_move = {int(v): k for k, v in self.move_vocab.items()}

        self.vocab_size = len(self.move_vocab)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fen = sample["fen"]
        move_uci = sample["move"]
        move_idx = self.move_vocab.get(move_uci)

        if move_idx is None:
            raise ValueError(f"Move {move_uci} not in move_vocab")

        # Tokenize FEN
        encoding = self.tokenizer(
            fen,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Compute legal move mask
        board = chess.Board(fen)
        legal_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for move in board.legal_moves:
            uci = move.uci()
            if uci in self.move_vocab:
                legal_mask[self.move_vocab[uci]] = True

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(move_idx, dtype=torch.long),
            "legal_mask": legal_mask
        }

# --- Training loop ---

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    dataset = ChessMoveDataset(args.data_path, args.move_vocab_path, tokenizer)
    print(f'Dataset loaded with {len(dataset)} samples and vocab size {dataset.vocab_size}')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=dataset.vocab_size
    ).to(device)

    
    
    print('Base model loaded.')
    print('Model details:',model)
    for param in model.distilbert.parameters():
        param.requires_grad = False


    optimizer = AdamW(model.parameters(), lr=args.lr)

    
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            legal_mask = batch["legal_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits  # shape: (batch_size, vocab_size)

            # Mask illegal moves
            mask_value = -1e9
            illegal_mask = ~legal_mask  # inverse mask
            logits = logits.masked_fill(illegal_mask, mask_value)

            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

# --- CLI entrypoint ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="magnus_finetune_for_bert.jsonl")
    parser.add_argument("--move_vocab_path", type=str, default="move_vocab.json")
    parser.add_argument("--output_dir", type=str, default="./bert_chess_model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    train(args)
