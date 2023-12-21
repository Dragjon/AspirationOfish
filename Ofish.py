import chess
import chess.polyglot
import chess.syzygy
import time

# Piece-square tables
piece_values_middlegame = {
    chess.PAWN: 82,
    chess.KNIGHT: 337,
    chess.BISHOP: 365,
    chess.ROOK: 477,
    chess.QUEEN: 1025,
    chess.KING: 0,
}

piece_values_endgame = {
    chess.PAWN: 94,
    chess.KNIGHT: 281,
    chess.BISHOP: 297,
    chess.ROOK: 512,
    chess.QUEEN:  936,
    chess.KING: 0,
}


piece_square_tables_middlegame = {
    chess.PAWN: [0, 0, 0, 0, 0, 0, 0, 0, -11, 34, 126, 68, 95, 61, 134, 98, -20, 25, 56, 65, 31, 26, 7, -6, -23, 17, 12, 23, 21, 6, 13, -14, -25, 10, 6, 17, 12, -5, -2, -27, -12, 33, 3, 3, -10, -4, -4, -26, -22, 38, 24, -15, -23, -20, -1, -35, 0, 0, 0, 0, 0, 0, 0, 0],

    chess.KNIGHT: [-107, -15, -97, 61, -49, -34, -89, -167, -17, 7, 62, 23, 36, 72, -41, -73, 44, 73, 129, 84, 65, 37, 60, -47, 22, 18, 69, 37, 53, 19, 17, -9, -8, 21, 19, 28, 13, 16, 4, -13, -16, 25, 17, 19, 10, 12, -9, -23, -19, -14, 18, -1, -3, -12, -53, -29, -23, -19, -28, -17, -33, -58, -21, -105],

    chess.BISHOP: [
        -8, 7, -42, -25, -37, -82, 4, -29,
        -47, 18, 59, 30, -13, -18, 16, -26,
        -2, 37, 50, 35, 40, 43, 37, -16,
        -2, 7, 37, 37, 50, 19, 5, -4,
        4, 10, 18, 34, 26, 19, 13, -6,
        10, 18, 27, 13, 12, 15, 15, 0,
        1, 33, 21, 7, 0, 16, 15, 4,
        -21, -39, -12, -13, -21, -14, -3, -33],

    chess.ROOK: [43, 31, 9, 63, 51, 32, 42, 32, 44, 26, 67, 80, 62, 58, 32, 27, 16, 61, 45, 17, 36, 26, 19, -5, -20, -8, 35, 24, 26, 7, -11, -24, -23, 6, -7, 9, -1, -12, -26, -36, -33, -5, 0, 3, -17, -16, -25, -45, -71, -6, 11, -1, -9, -20, -16, -44, -26, -37, 7, 16, 17, 1, -13, -19],

    chess.QUEEN: [45, 43, 44, 59, 12, 29, 0, -28, 54, 28, 57, -16, 1, -5, -39, -24, 57, 47, 56, 29, 8, 7, -17, -13, 1, -2, 17, -1, -16, -16, -27, -27, -3, 3, -4, -2, -10, -9, -26, -9, 5, 14, 2, -5, -2, -11, 2, -14, 1, -3, 15, 8, 2, 11, -8, -35, -50, -31, -25, -15, 10, -9, -18, -1],

    chess.KING: [13, 2, -34, -56, -15, 16, 23, -65, -29, -38, -4, -8, -7, -20, -1, 29, -22, 22, 6, -20, -16, 2, 24, -9, -36, -14, -25, -30, -27, -12, -20, -17, -51, -33, -44, -46, -39, -27, -1, -49, -27, -15, -30, -44, -46, -22, -14, -14, 8, 9, -16, -43, -64, -8, 7, 1, 14, 24, -28, 8, -54, 12, 36, -15]


}

piece_square_tables_endgame = {
    chess.PAWN: [0, 0, 0, 0, 0, 0, 0, 0, 187, 165, 132, 147, 134, 158, 173, 178, 84, 82, 53, 56, 67, 85, 100, 94, 17, 17, 4, -2, 5, 13, 24, 32, -1, 3, -8, -7, -7, -3, 9, 13, -8, -1, -5, 0, 1, -6, 7, 4, -7, 2, 0, 13, 10, 8, 8, 13, 0, 0, 0, 0, 0, 0, 0, 0],

    chess.KNIGHT: [-99, -63, -27, -31, -28, -13, -38, -58, -52, -24, -25, -9, -2, -25, -8, -25, -41, -19, -9, -1, 9, 10, -20, -24, -18, 8, 11, 22, 22, 22, 3, -17, -18, 4, 17, 16, 25, 16, -6, -18, -22, -20, -3, 10, 15, -1, -3, -23, -44, -23, -20, -2, -5, -10, -20, -42, -64, -50, -18, -22, -15, -23, -51, -29],

    chess.BISHOP: [-24, -17, -9, -7, -8, -11, -21, -14, -14, -4, -13, -3, -12, 7, -4, -8, 4, 0, 6, -2, -1, 0, -8, 2, 2, 3, 10, 14, 9, 12, 9, -3, -9, -3, 10, 7, 19, 13, 3, -6, -15, -7, 3, 13, 10, 8, -3, -12, -27, -15, -9, 4, -1, -7, -18, -14, -17, -5, -16, -9, -5, -23, -9, -23],

    chess.ROOK: [5, 8, 12, 12, 15, 18, 10, 13, 3, 8, 3, -3, 11, 13, 13, 11, -3, -5, -3, 4, 5, 7, 7, 7, 2, -1, 1, 2, 1, 13, 3, 4, -11, -8, -6, -5, 4, 8, 5, 3, -16, -8, -12, -7, -1, -5, 0, -4, -3, -11, -9, -9, 2, 0, -6, -6, -20, 4, -13, -5, -1, 3, 2, -9],

    chess.QUEEN: [20, 10, 19, 27, 27, 22, 22, -9, 0, 30, 25, 58, 41, 32, 20, -17, 9, 19, 35, 47, 49, 9, 6, -20, 36, 57, 40, 57, 45, 24, 22, 3, 23, 39, 34, 31, 47, 19, 28, -18, 5, 10, 17, 9, 6, 15, -27, -16, -32, -36, -23, -16, -16, -30, -23, -22, -41, -20, -32, -5, -43, -22, -28, -33],

    chess.KING: [-17, 4, 15, -11, -18, -18, -35, -74, 11, 23, 38, 17, 17, 14, 17, -12, 13, 44, 45, 20, 15, 23, 17, 10, 3, 26, 33, 26, 27, 24, 22, -8, -11, 9, 23, 27, 24, 21, -4, -18, -9, 7, 16, 23, 21, 11, -3, -19, -17, -5, 4, 14, 13, 4, -11, -27, -43, -24, -14, -28, -11, -21, -34, -53]



}

ENDGAME_OPP_KING_CORNERED_TABLE = [
    100,   95,   90,   85,   85,   90, 95,   100,
    95,   60,   50,   50,   50,   50,  60,   95,
    90,   50,   10,   10,   10,   10,  50,   90,
    85,   50,   10,   -20,   -20,   10,  50,   85,
    85,   50,   10,   -20,   -20,   10,  50,   85,
    90,   50,   10,   10,   10,   10,  50,   90,
    95,   60,   50,   50,   50,   50,  60,   95,
    100,   95,   90,   85,   85,   90,  95,   100,


]




def is_endgame_position(board):
  # Check if there are no major pieces
  no_major_pieces = (
      sum(1 for _ in board.pieces(chess.ROOK, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.QUEEN, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.ROOK, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.QUEEN, chess.BLACK)) == 0)
  if no_major_pieces:
    return True

  # Check for two rooks and no queen
  if (sum(1 for _ in board.pieces(chess.QUEEN, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.ROOK, chess.WHITE)) == 1
      and sum(1 for _ in board.pieces(chess.QUEEN, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.ROOK, chess.BLACK)) == 1):
    return True

  # Check for two queens, kings, and no other pieces
  if (sum(1 for _ in board.pieces(chess.QUEEN, chess.WHITE)) <= 1
      and sum(1 for _ in board.pieces(chess.KING, chess.WHITE)) == 1
      and sum(1 for _ in board.pieces(chess.KNIGHT, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.BISHOP, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.ROOK, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.PAWN, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.QUEEN, chess.BLACK)) <= 1
      and sum(1 for _ in board.pieces(chess.KING, chess.BLACK)) == 1
      and sum(1 for _ in board.pieces(chess.KNIGHT, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.BISHOP, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.ROOK, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.PAWN, chess.BLACK)) == 0):
    return True

  # If none of the above conditions are met, return False
  return False


def evaluate_board(board):

    total_evaluation = 0

    # If there are 5 or fewer pieces on the board, probe the tablebase for the best move
    if len(board.piece_map()) <= 5:
        try:
            wdl = tablebases.probe_wdl(board)
            dtz = tablebases.probe_dtz(board)
            if wdl is not None and dtz is not None:
              if board.turn == chess.WHITE:
                if wdl == 2:
                  return 10000 - dtz
                elif wdl == -2:
                  return -10000 + dtz
                else:
                  return 0

              else:
                if wdl == 2:
                  return -10000 + dtz
                elif wdl == -2:
                  return 10000 - dtz
                else:
                  return 0
        except KeyError:
            pass

    if not is_endgame_position(board):
        piece_square_tables = piece_square_tables_middlegame
        piece_values = piece_values_middlegame
    else:
        piece_square_tables = piece_square_tables_endgame
        piece_values = piece_values_endgame

    # Evaluate material and position
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            color = board.piece_at(square).color
            piece_type = board.piece_at(square).piece_type

            if color == chess.WHITE:
                square_value = piece_square_tables[piece_type][63 - square]
            else:
                square_value = piece_square_tables[piece_type][square]

            total_evaluation += piece_values[piece_type] if color == chess.WHITE else -piece_values[piece_type]
            total_evaluation += square_value if color == chess.WHITE else -square_value

            # Penalize friendly pieces in front of pawns
            if piece_type == chess.PAWN:
                pawn_front_square = chess.square(chess.square_file(square), chess.square_rank(square) + 1 if color == chess.WHITE else chess.square_rank(square) - 1)
                if board.piece_at(pawn_front_square) is not None and board.piece_at(pawn_front_square).color == color:
                    total_evaluation -= 5  # Adjust the penalty value as needed


    if is_endgame_position(board):
        # Bonus for negating the opponent's king mobility
        opponent_king_square = board.king(chess.BLACK if board.turn == chess.WHITE else chess.WHITE)
        opponent_king_mobility = len(board.attackers(chess.WHITE, opponent_king_square) | board.attackers(chess.BLACK, opponent_king_square))
        total_evaluation -= 35 * opponent_king_mobility  # You can adjust the bonus value

        # King proximity bonus
        own_king_square = board.king(board.turn)
        distance_to_opponent_king = chess.square_distance(own_king_square, opponent_king_square)
        total_evaluation += 35 * distance_to_opponent_king  # You can adjust the bonus value

        total_evaluation += ENDGAME_OPP_KING_CORNERED_TABLE[opponent_king_square]

    return total_evaluation

def evaluate_move(move, board):
  move = board.uci(move)
  movebonus = 0
  if (move == "e1g1") or (move == "e1c1"):
    movebonus += 5
  if (move == "e8g8") or (move == "e8c8"):
    movebonus -= 5
  return movebonus



def is_king_and_rook_endgame(board):
  # Check for 1 rook, kings, and no other pieces
  if (sum(1 for _ in board.pieces(chess.QUEEN, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.KING, chess.WHITE)) == 1
      and sum(1 for _ in board.pieces(chess.KNIGHT, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.BISHOP, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.ROOK, chess.WHITE)) == 1
      and sum(1 for _ in board.pieces(chess.PAWN, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.QUEEN, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.KING, chess.BLACK)) == 1
      and sum(1 for _ in board.pieces(chess.KNIGHT, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.BISHOP, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.ROOK, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.PAWN, chess.BLACK)) == 0):
    return True

  # Check for 1 rook, kings, and no other pieces
  if (sum(1 for _ in board.pieces(chess.QUEEN, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.KING, chess.WHITE)) == 1
      and sum(1 for _ in board.pieces(chess.KNIGHT, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.BISHOP, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.ROOK, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.PAWN, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.QUEEN, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.KING, chess.BLACK)) == 1
      and sum(1 for _ in board.pieces(chess.KNIGHT, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.BISHOP, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.ROOK, chess.BLACK)) == 1
      and sum(1 for _ in board.pieces(chess.PAWN, chess.BLACK)) == 0):
    return True

  return False


def is_king_and_queen_endgame(board):
  # Check for 1 rook, kings, and no other pieces
  if (sum(1 for _ in board.pieces(chess.QUEEN, chess.WHITE)) == 1
      and sum(1 for _ in board.pieces(chess.KING, chess.WHITE)) == 1
      and sum(1 for _ in board.pieces(chess.KNIGHT, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.BISHOP, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.ROOK, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.PAWN, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.QUEEN, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.KING, chess.BLACK)) == 1
      and sum(1 for _ in board.pieces(chess.KNIGHT, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.BISHOP, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.ROOK, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.PAWN, chess.BLACK)) == 0):
    return True

  # Check for 1 rook, kings, and no other pieces
  if (sum(1 for _ in board.pieces(chess.QUEEN, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.KING, chess.WHITE)) == 1
      and sum(1 for _ in board.pieces(chess.KNIGHT, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.BISHOP, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.ROOK, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.PAWN, chess.WHITE)) == 0
      and sum(1 for _ in board.pieces(chess.QUEEN, chess.BLACK)) == 1
      and sum(1 for _ in board.pieces(chess.KING, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.KNIGHT, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.BISHOP, chess.BLACK)) == 0
      and sum(1 for _ in board.pieces(chess.ROOK, chess.BLACK)) == 1
      and sum(1 for _ in board.pieces(chess.PAWN, chess.BLACK)) == 0):
    return True

  return False


def quiescence(board, alpha, beta, color, depth, ply):
  if depth == 0 or board.is_game_over():
      if board.can_claim_draw() or board.is_stalemate() or board.is_insufficient_material():
            return 0

      if board.is_checkmate():
          return -10000+ply

  stand_pat = color * evaluate_board(board)
  if stand_pat >= beta:
    return beta
  if alpha < stand_pat:
    alpha = stand_pat

  legal_captures = [
      move for move in board.legal_moves if board.is_capture(move)
  ]

  for move in legal_captures:
    bonus = evaluate_move(move, board)
    board.push(move)
    score = -quiescence(board, -beta, -alpha, -color, depth - 1, ply+1) - bonus
    board.pop()

    if score >= beta:
      return beta
    if score > alpha:
      alpha = score

  # Beta cutoff, mistakenly called "delta pruning" by chat gpt
  delta = beta - alpha
  if delta <= 0:
    return alpha 

  return alpha


def negamax_alpha_beta(board, depth, alpha, beta, color, nodes, start_time, ply):

    elapsed_time = time.time() - start_time
    hard_limit = start_time / 30

    if (elapsed_time > hard_limit) and (ply != 1):
      return None, None
    

    if depth == 0 or board.is_game_over():
        if board.can_claim_draw() or board.is_stalemate() or board.is_insufficient_material():
            return 0, nodes

        if board.is_checkmate():
            return -10000+ply, nodes

        return quiescence(board, alpha, beta, color, 3, ply+1), nodes

    syzygy_max = float('inf')
    syzygy_min = float('-inf')

    if len(board.piece_map()) <= 5: # hardcoded 5?
        wdl = tablebases.probe_wdl(board)
        if wdl is not None:
            if wdl == -2: # loss
                tb_score = -9000
                if tb_score <= alpha:
                    return tb_score, nodes
                syzygy_max = tb_score
            elif wdl == 2: # win
                tb_score = 9000
                if tb_score >= beta:
                    return tb_score, nodes
                syzygy_min = tb_score
            else: # draw, cursed win or blessed loss
                return 0, nodes

    legal_moves = list(board.legal_moves)

    # Sort moves based on MVV-LVA ordering
    moves = sorted(legal_moves,
                   key=lambda move: mvv_lva_ordering(move, board),
                   reverse=True)

    for move in moves:
        bonus = evaluate_move(move, board)
        board.push(move)
        nodes += 1
        value, nodes = negamax_alpha_beta(board, depth - 1, -beta, -alpha, -color, nodes, start_time, ply+1)

        value = -value - bonus
        board.pop()
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    if alpha < syzygy_min:
        alpha = syzygy_min
    if alpha > syzygy_max:
        alpha = syzygy_max

    return alpha, nodes

def get_best_move(board, depth, start_time, nodes):
    best_move = None
    best_value = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    legal_moves = list(board.legal_moves)
    elapsed_time = time.time() - start_time
    hard_limit = start_time / 30
    ply = 0

    if (elapsed_time > hard_limit) and (depth != 1):
        return None, None

    # Try moves from the opening book first
    with chess.polyglot.open_reader(opening_book_path) as reader:
        for entry in reader.find_all(board):
            move = entry.move
            if move in legal_moves:
                return move, 0, nodes  # Return a move from the opening book

    if board.turn == chess.WHITE:
        color = 1
    else:
        color = -1

    # Sort moves based on some heuristic for move ordering
    moves = sorted(legal_moves,
                   key=lambda move: mvv_lva_ordering(move, board),
                   reverse=True)

    for move in moves:
        board.push(move)
        nodes += 1
        if board.is_checkmate():
            return move, 10000, nodes
        value, nodes = negamax_alpha_beta(board, depth - 1, -beta, -alpha, -color, nodes, start_time, ply+1)
        value = -value
        board.pop()

        if value > best_value:
            best_value = value
            best_move = move

        alpha = max(alpha, value)

    return best_move, best_value, nodes

def mvv_lva_ordering(move, board):
  # Most Valuable Victim - Least Valuable Aggressor ordering
  capturing_piece = board.piece_at(move.to_square)
  victim_piece = board.piece_at(move.from_square)

  if capturing_piece and victim_piece:
    return static_piece_value(capturing_piece) - static_piece_value(
        victim_piece)
  else:
    return 0


def static_piece_value(piece):
  # Assign values to pieces for MVV-LVA ordering
  if piece.piece_type == chess.PAWN:
    return 90
  elif piece.piece_type == chess.KNIGHT:
    return 320
  elif piece.piece_type == chess.BISHOP:
    return 325
  elif piece.piece_type == chess.ROOK:
    return 500
  elif piece.piece_type == chess.QUEEN:
    return 900
  elif piece.piece_type == chess.KING:
    return 600
  else:
    return 0

# Specify the path to the endgame tablebase files
tablebase_path = "./tablebase_files"
tablebases = chess.syzygy.open_tablebase(tablebase_path)


def calculateMaxTime(board, remaining_time):
    return remaining_time / 60

contempt = 0
usebook = True
opening_book_path = "komodo.bin"

def main():
    global contempt, usebook, opening_book_path
    board = chess.Board()

    uci_mode = False
    wtime = 1000000
    btime = 1000000
    remainingtime = 1000000

    while True:
        input_line = input()
        if input_line == "uci":
            print("id name OfishV1K")
            print("id author Chess123easy")
            print("option name Contempt type spin default 0 min -100 max 100")
            print("option name UseBook type check default true")
            print(f"option name BookPath type string default {opening_book_path}")
            print("uciok")
            uci_mode = True
        elif input_line == "isready":
            print("readyok")
        elif input_line.startswith("position"):
            parts = input_line.split()
            if len(parts) < 2:
                continue
            position_type = parts[1]
            if position_type == "startpos":
                board.set_fen(chess.STARTING_FEN)
                if len(parts) > 2 and parts[2] == "moves":
                    for move in parts[3:]:
                        board.push_uci(move)
            elif position_type == "fen":
                if len(parts) < 8:
                    continue
                fen = " ".join(parts[2:8])
                board.set_fen(fen)
                if len(parts) > 8 and parts[8] == "moves":
                    for move in parts[9:]:
                        board.push_uci(move)
            position_fen = board.fen()

        elif input_line.startswith("go"):
            if not uci_mode:
                continue

            # Parse additional parameters for search
            parameters = input_line.split()[1:]
            max_time = 0  # Set a default maximum time
            max_depth = 0  # Set a default maximum depth

            for i in range(len(parameters)):
                if parameters[i] == "wtime" and i + 1 < len(parameters):
                    wtime = float(parameters[i + 1])
                elif parameters[i] == "btime" and i + 1 < len(parameters):
                    btime = float(parameters[i + 1])
                elif parameters[i] == "contempt" and i + 1 < len(parameters):
                    contempt = int(parameters[i + 1])
                elif parameters[i] == "usebook" and i + 1 < len(parameters):
                    usebook = parameters[i + 1].lower() == "true"
                elif parameters[i] == "bookpath" and i + 1 < len(parameters):
                    opening_book_path = parameters[i + 1]

            remainingtime = wtime / 1000 if board.turn == chess.WHITE else btime / 1000

            start_time = time.time()  # Start the timer

            depth = 1  # Start with depth 1

            # Inside the while loop in uci_loop function
            while depth <= 4:

                # Check if the maximum time has been exceeded
                elapsed_time = time.time() - start_time
                if elapsed_time > calculateMaxTime(board, remainingtime):
                    break

                nodes = 0
                best_movex, scorex, nodes = get_best_move(board, depth, start_time, nodes)

                if (best_movex == None) and (scorex == None):
                    break
                if (scorex == 10000):
                    best_move, score = best_movex, scorex
                    break
                else:
                    best_move, score = best_movex, scorex

                print(
                    f"info depth {depth} score cp {score} nodes {nodes} time {int(elapsed_time * 1000)} nps {int(nodes/elapsed_time)} pv {best_move}"
                )

                # Increase the search depth for the next iteration
                depth += 1

            # Output the final result
            print("bestmove", best_move.uci())

        elif input_line == "quit":
            break

if __name__ == "__main__":
    main()
