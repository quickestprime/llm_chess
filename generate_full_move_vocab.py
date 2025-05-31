import chess
import json

def generate_full_uci_move_vocab():
    move_set = set()

    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            # Skip same square
            if from_square == to_square:
                continue

            # Try base move
            try:
                move = chess.Move(from_square, to_square)
                uci = move.uci()
                move_set.add(uci)
            except:
                pass

            # Try promotions (only on 2nd to 1st rank for black, 7th to 8th for white)
            from_rank = chess.square_rank(from_square)
            to_rank = chess.square_rank(to_square)

            is_promotion = (
                (from_rank == 6 and to_rank == 7) or  # white promotes
                (from_rank == 1 and to_rank == 0)     # black promotes
            )

            if is_promotion:
                for promo in ['q', 'r', 'b', 'n']:
                    try:
                        promo_move = chess.Move(from_square, to_square, promotion=chess.Piece.from_symbol(promo).piece_type)
                        uci = promo_move.uci()
                        move_set.add(uci)
                    except:
                        pass

    # Add castling explicitly
    castling_moves = ["e1g1", "e1c1", "e8g8", "e8c8"]
    move_set.update(castling_moves)

    # Sort and index
    sorted_moves = sorted(move_set)
    move_vocab = {uci: idx for idx, uci in enumerate(sorted_moves)}

    # Save
    with open("move_vocab.json", "w") as f:
        json.dump(move_vocab, f, indent=2)

    print(f"Generated move_vocab.json with {len(move_vocab)} moves.")

if __name__ == "__main__":
    generate_full_uci_move_vocab()
