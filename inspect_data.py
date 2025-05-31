import chess.pgn

# Open the PGN file
with open("Carlsen.pgn", "r", encoding="utf-8") as pgn_file:
    game_num = 1

    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break  # Reached end of file

        print(f"\n=== Game {game_num} ===")
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        result = game.headers.get("Result", "?")

        print(f"White: {white}")
        print(f"Black: {black}")
        print(f"Result: {result}")

        # See if Carlsen played and which color
        if "Carlsen" in white:
            print("Magnus played White")
        elif "Carlsen" in black:
            print("Magnus played Black")
        else:
            print("Magnus did not play this game?")
            continue

        board = game.board()
        node = game

        print("\nFirst 5 moves:")
        for i, move in enumerate(game.mainline_moves()):
            if i >= 5:
                break

            move_san = board.san(move)
            fen_before = board.fen()
            board.push(move)
            fen_after = board.fen()

            print(f"Move {i + 1}: {move_san}")
            print(f"  FEN before: {fen_before}")
            print(f"  FEN after : {fen_after}")

        game_num += 1

        # Limit to first few games for now
        if game_num > 3:
            break
