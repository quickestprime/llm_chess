import chess
import random
from tqdm import trange
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import json
from pathlib import Path
from stockfish import Stockfish
import time
import numpy as np

model_dir = Path(__file__).parent / "bert_chess_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
print("Loaded model.")


stockfish_elo = 1
stockfish = Stockfish()
stockfish.set_elo_rating(stockfish_elo)  # Set Stockfish elo here
print(f"Loaded stockfish and set ELO to {stockfish_elo}.")

with open("move_vocab.json") as f:
    move_vocab = json.load(f)

print(f"Loaded move vocab with {len(move_vocab)} entries.")
idx_to_move = {str(v): k for k, v in move_vocab.items()}

def get_bot_move(fen):
    
    board = chess.Board(fen)
    encoding = tokenizer(fen, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    
    with torch.no_grad():
        logits = model(**encoding).logits.squeeze()

    legal_mask = torch.zeros(logits.size(0), dtype=torch.bool)
    for move in board.legal_moves:
        uci = move.uci()
        if uci in move_vocab:
            legal_mask[move_vocab[uci]] = True
    
    logits[~legal_mask] = -1.0 * np.inf

    move_idx = torch.argmax(logits).item()
    move_uci = idx_to_move.get(str(move_idx))
    return move_uci 

def play_game(engine, play_white=True, verbose=True):
    board = chess.Board()
    move_history = []

    while not board.is_game_over():
        
        if (board.turn == chess.WHITE and play_white) or (board.turn == chess.BLACK and not play_white):
            # it's our turn
            visualise = random.random() < 0.1
            if visualise:
                print(stockfish.get_board_visual(play_white))
            move_uci = get_bot_move(board.fen())
            board.push_uci(move_uci)
            move_history.append(("Bot", move_uci))
            if visualise:
                print(stockfish.get_board_visual(play_white))

        else:
            engine.set_fen_position(board.fen())
            sf_move = engine.get_best_move_time(100)  # 100 ms think time
            board.push_uci(sf_move)
            move_history.append(("Stockfish", sf_move))

    result = board.result()
    total_plies = len(move_history)
    full_moves = total_plies // 2 + total_plies % 2

    if verbose:
        print(f"\nGame over in {full_moves} moves ({total_plies} plies). Result: {result}")
        if result == "1-0":
            print("Winner: White (Bot)" if play_white else "Winner: White (Stockfish)")
        elif result == "0-1":
            print("Winner: Black (Bot)" if not play_white else "Winner: Black (Stockfish)")
        else:
            print("Result: Draw")

    if result == "1-0":
        return 1 if play_white else 0
    elif result == "0-1":
        return 0 if play_white else 1
    else:
        return 0.5


def estimate_elo(score, opponent_elo):
    if score == 0:
        return 0
    return opponent_elo + 400 * np.log10(score / (1 - score))

def main():
    games = 2

    total_score = 0
    for i in trange(games):
        print("Playing game {}".format(i+1))
        play_white = i % 2 == 0
        score = play_game(stockfish, play_white)

    avg_score = total_score / games
    est_elo = estimate_elo(avg_score, stockfish_elo)

    print(f"\nBot scored {avg_score * 100:.1f}% vs Stockfish ({stockfish_elo} Elo)")
    print(f"Estimated Elo: {est_elo:.0f}")

if __name__ == "__main__":
    main()
