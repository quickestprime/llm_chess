import chess.pgn
import json

input_pgn_path = "Carlsen.pgn"
output_jsonl_path = "chess_finetune_simple.jsonl"

def get_piece_index(move, board):
    """
    Encode the moving piece as an index from 0–15:
    file (0–7) × piece type (0–1):
    0 = Rook, 1 = Knight, 2 = Bishop, 3 = Queen, 4 = King, 5 = Pawn
    => piece_type_idx × 8 + file
    """
    from_square = move.from_square
    file = chess.square_file(from_square)
    piece = board.piece_at(from_square)
    
    if piece is None:
        return None  # Shouldn't happen

    type_order = {
        chess.ROOK: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.QUEEN: 3,
        chess.KING: 4,
        chess.PAWN: 5
    }

    type_idx = type_order.get(piece.piece_type)
    if type_idx is None:
        return None

    return type_idx * 8 + file  # 0–47

def get_square_index(move):
    """
    Encode destination square as 0–63: a1 = 0, b1 = 1, ..., h8 = 63
    """
    return move.to_square

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

                piece_idx = get_piece_index(move, board)
                square_idx = get_square_index(move)

                if piece_idx is not None:
                    samples.append({
                        "fen": fen_before,
                        "piece_label": piece_idx,
                        "square_label": square_idx
                    })

            board.push(move)
            node = next_node

# Save to JSONL
with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for sample in samples:
        json.dump(sample, f)
        f.write("\n")

print(f"Saved {len(samples)} samples to {output_jsonl_path}")
