import chess
import random
from tqdm import trange
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import json
from pathlib import Path
from stockfish import Stockfish
import numpy as np

model_dir = Path(__file__).parent / "bert_chess_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
model.eval()
print("Loaded model.")

stockfish_elo = 1500  # You can set Stockfish elo here for testing
stockfish = Stockfish()
stockfish.set_elo_rating(stockfish_elo)
stockfish.set_skill_level(20)  # Max skill, can reduce for easier opponent
print(f"Loaded Stockfish with Elo set to {stockfish_elo}.")

with open("move_vocab.json") as f:
    move_vocab = json.load(f)

print(f"Loaded move vocab with {len(move_vocab)} entries.")
idx_to_move = {str(v): k for k, v in move_vocab.items()}


def get_bot_move(fen):
    board = chess.Board(fen)
    encoding = tokenizer(fen, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    
    with torch.no_grad():
        logits = model(**encoding).logits.squeeze()

    legal_mask = torch.zeros_like(logits, dtype=torch.bool)
    for move in board.legal_moves:
        uci = move.uci()
        if uci in move_vocab:
            legal_mask[move_vocab[uci]] = True

    # Mask out illegal moves to -inf logits
    logits[~legal_mask] = -float('inf')

    if (legal_mask.sum() == 0).item():
        # No legal moves predicted in vocab (shouldn't happen), fallback to random legal move
        return random.choice([m.uci() for m in board.legal_moves])

    move_idx = torch.argmax(logits).item()
    move_uci = idx_to_move.get(str(move_idx))

    # Safety check: fallback if move_uci not legal (can happen if vocab incomplete)
    if move_uci not in [m.uci() for m in board.legal_moves]:
        move_uci = random.choice([m.uci() for m in board.legal_moves])
    return move_uci


def play_game(engine, play_white=True, verbose=True):
    board = chess.Board()
    move_history = []

    while not board.is_game_over():
        if (board.turn == chess.WHITE and play_white) or (board.turn == chess.BLACK and not play_white):
            # Bot's turn
            move_uci = get_bot_move(board.fen())
            board.push_uci(move_uci)
            move_history.append(("Bot", move_uci))
            if verbose and random.random() < 0.1:
                print(f"Bot plays: {move_uci}")
        else:
            # Stockfish's turn
            engine.set_fen_position(board.fen())
            sf_move = engine.get_best_move_time(100)  # 100 ms think time
            board.push_uci(sf_move)
            move_history.append(("Stockfish", sf_move))
            if verbose and random.random() < 0.1:
                print(f"Stockfish plays: {sf_move}")

    result = board.result()
    total_plies = len(move_history)
    full_moves = total_plies // 2 + total_plies % 2

    if verbose:
        print(f"\nGame over in {full_moves} moves ({total_plies} plies). Result: {result}")
        if result == "1-0":
            winner = "White (Bot)" if play_white else "White (Stockfish)"
        elif result == "0-1":
            winner = "Black (Bot)" if not play_white else "Black (Stockfish)"
        else:
            winner = "Draw"
        print(f"Winner: {winner}")

    # Return bot's score from 0 to 1
    if result == "1-0":
        return 1.0 if play_white else 0.0
    elif result == "0-1":
        return 0.0 if play_white else 1.0
    else:
        return 0.5


def estimate_elo(score, opponent_elo):
    """
    Elo estimation from score vs opponent Elo.
    Handle edge cases by bounding score away from 0 and 1.
    """
    epsilon = 1e-5
    score = min(max(score, epsilon), 1 - epsilon)
    expected_score = score
    elo_diff = -400 * np.log10(1 / expected_score - 1)
    return opponent_elo + elo_diff


def main():
    games = 10  # Number of games to play (increase for better estimate)

    total_score = 0
    for i in trange(games):
        print(f"\nPlaying game {i+1}/{games}")
        play_white = (i % 2 == 0)  # Alternate colors
        score = play_game(stockfish, play_white, verbose=True)
        total_score += score

    avg_score = total_score / games
    est_elo = estimate_elo(avg_score, stockfish_elo)

    print(f"\nBot scored {avg_score * 100:.1f}% vs Stockfish (Elo {stockfish_elo})")
    print(f"Estimated Elo: {est_elo:.0f}")


if __name__ == "__main__":
    main()
