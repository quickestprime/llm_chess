import chess.pgn
import json

input_pgn_path = "Carlsen.pgn"
output_jsonl_path = "magnus_finetune_for_bert.jsonl"
move_vocab_path = "move_vocab.json"

# Load existing move_vocab from file
with open(move_vocab_path, "r", encoding="utf-8") as f:
    move_vocab = json.load(f)

# Reverse lookup: move string → index is move_vocab already, 
# but we want to keep track of highest index used in case needed
# (not strictly needed if you only want to use existing vocab)
max_index = max(move_vocab.values()) if move_vocab else -1

def get_move_index(move_str):
    # Instead of adding new moves, just lookup existing vocab
    # If move_str not found, raise error or skip sample to avoid misalignment
    if move_str not in move_vocab:
        raise ValueError(f"Move '{move_str}' not found in move_vocab.json")
    return move_vocab[move_str]

samples = []

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
                try:
                    move_idx = get_move_index(move_str)
                except ValueError:
                    # Skip moves not in vocab to avoid misalignment
                    node = next_node
                    continue

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

print(f"Saved {len(samples)} training samples to {output_jsonl_path}")
print(f"Using move vocabulary of size {len(move_vocab)}")
