import chess
import chess.engine
import random
import subprocess
from tqdm import trange
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import json

# Load model
tokenizer = BertTokenizerFast.from_pretrained("./bert_chess_model")
model = BertForSequenceClassification.from_pretrained("./bert_chess_model")
model.eval()

with open("move_vocab.json") as f:
    move_vocab = json.load(f)
idx_to_move = {v: k for k, v in move_vocab.items()}

def get_bot_move(fen):
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
    logits[~legal_mask] = -1e9

    move_idx = torch.argmax(logits).item()
    move_uci = idx_to_move.get(str(move_idx))
    return move_uci if move_uci and chess.Move.from_uci(move_uci) in board.legal_moves else None

def play_game(engine, elo=1600, play_white=True):
    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.WHITE and play_white or board.turn == chess.BLACK and not play_white:
            # MagnusBot's turn
            move_uci = get_bot_move(board.fen())
            if move_uci is None:
                break  # No legal move predicted
            board.push_uci(move_uci)
        else:
            # Stockfish's turn
            result = engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)

    result = board.result()
    if result == "1-0":
        return 1 if play_white else 0
    elif result == "0-1":
        return 0 if play_white else 1
    else:
        return 0.5

def estimate_elo(score, opponent_elo):
    import numpy as np
    return opponent_elo + 400 * np.log10(score / (1 - score))

def main():
    games = 50
    stockfish_path = "stockfish"  # Adjust path if needed
    target_elo = 1600

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": target_elo})
        total_score = 0
        for i in trange(games):
            play_white = i % 2 == 0
            score = play_game(engine, target_elo, play_white)
            total_score += score

    avg_score = total_score / games
    est_elo = estimate_elo(avg_score, target_elo)

    print(f"\nBot scored {avg_score * 100:.1f}% vs Stockfish ({target_elo} Elo)")
    print(f"Estimated Elo: {est_elo:.0f}")

if __name__ == "__main__":
    main()
