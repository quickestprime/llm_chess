import chess.pgn
import json
from collections import defaultdict

# https://www.pgnmentor.com/files.html
input_pgn_path = "Carlsen.pgn"
output_jsonl_path = "magnus_finetune_for_bert.jsonl"

samples = []
move_vocab = {}  # UCI move string → index
move_counter = 0

def get_move_index(move_str):
    global move_counter
    if move_str not in move_vocab:
        move_vocab[move_str] = move_counter
        move_counter += 1
    return move_vocab[move_str]

with open(input_pgn_path, "r", encoding="utf-8") as pgn_file:
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break

        board = game.board()
        node = game

        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")

        while not node.is_end():
            next_node = node.variation(0)
            move = next_node.move
            player_name = white if board.turn == chess.WHITE else black

            if "Carlsen" in player_name:
                fen_before = board.fen()
                board.push(move)

                move_str = move.uci()
                move_idx = get_move_index(move_str)

                samples.append({
                    "fen": fen_before,
                    "move": move_str,
                    "move_idx": move_idx
                })
            else:
                board.push(move)

            node = next_node

# Save training samples
with open(output_jsonl_path, "w", encoding="utf-8") as out_file:
    for sample in samples:
        json.dump(sample, out_file)
        out_file.write("\n")

# Optionally save move vocab
with open("move_vocab.json", "w") as f:
    json.dump(move_vocab, f, indent=2)

print(f"Saved {len(samples)} training samples to {output_jsonl_path}")
print(f"Move vocabulary has {len(move_vocab)} unique UCI moves.")
