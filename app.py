# app.py

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import chess
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
import json

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model + tokenizer + move vocab
tokenizer = DistilBertTokenizerFast.from_pretrained("./bert_chess_model")
model = DistilBertForSequenceClassification.from_pretrained("./bert_chess_model")
model.eval()

with open("move_vocab.json") as f:
    move_vocab = json.load(f)
idx_to_move = {v: k for k, v in move_vocab.items()}

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html") as f:
        return f.read()

@app.post("/move")
async def get_bot_move(request: Request):
    data = await request.json()
    fen = data["fen"]

    board = chess.Board(fen)
    encoding = tokenizer(fen, return_tensors="pt", truncation=True, padding='max_length', max_length=128)

    with torch.no_grad():
        logits = model(**encoding).logits.squeeze()

    # Mask illegal moves
    legal_mask = torch.zeros(logits.size(0), dtype=torch.bool)
    for move in board.legal_moves:
        uci = move.uci()
        if uci in move_vocab:
            legal_mask[move_vocab[uci]] = True
    logits[~legal_mask] = -1e9  # mask illegal

    move_idx = torch.argmax(logits).item()
    move_uci = idx_to_move.get(str(move_idx))

    if move_uci is None or chess.Move.from_uci(move_uci) not in board.legal_moves:
        return JSONResponse({"error": "No valid move found"}, status_code=400)

    return {"move": move_uci}
