import chess.pgn
import json

input_pgn_path = "Carlsen.pgn"
output_jsonl_path = "magnus_piece_dest_train.jsonl"

# Map chess piece symbols to piece type classes 0-5
# Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4, King=5
piece_type_map = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

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

            # Only keep moves by Magnus Carlsen (adjust as needed)
            player_name = white if board.turn == chess.WHITE else black
            if "Carlsen" in player_name:
                fen_before = board.fen()
                piece_moved = board.piece_at(move.from_square)
                if piece_moved is None:
                    # Skip illegal or corrupted data
                    board.push(move)
                    node = next_node
                    continue

                piece_type = piece_type_map[piece_moved.piece_type]
                destination_square = move.to_square  # 0-63

                samples.append({
                    "fen": fen_before,
                    "piece_type": piece_type,
                    "destination_square": destination_square
                })

            board.push(move)
            node = next_node

print(f"Collected {len(samples)} training samples.")

# Save samples as JSONL
with open(output_jsonl_path, "w", encoding="utf-8") as out_file:
    for sample in samples:
        json.dump(sample, out_file)
        out_file.write("\n")

print(f"Saved training data to {output_jsonl_path}")
