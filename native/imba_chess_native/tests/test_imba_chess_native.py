"""Comprehensive tests for the cozy-chess Python wrapper."""
import copy
import pytest

import imba_chess_native


# ═══════════════════════════════════════════════════════════════════════════
# Enum Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestColor:
    def test_variants(self):
        assert imba_chess_native.Color.White != imba_chess_native.Color.Black

    def test_invert(self):
        assert ~imba_chess_native.Color.White == imba_chess_native.Color.Black
        assert ~imba_chess_native.Color.Black == imba_chess_native.Color.White

    def test_str(self):
        # Rust Display format uses lowercase single letter
        assert str(imba_chess_native.Color.White) in ("w", "W", "White", "white")
        assert str(imba_chess_native.Color.Black) in ("b", "B", "Black", "black")

    def test_hash(self):
        d = {imba_chess_native.Color.White: "w", imba_chess_native.Color.Black: "b"}
        assert d[imba_chess_native.Color.White] == "w"


class TestPiece:
    def test_all_variants(self):
        pieces = [
            imba_chess_native.Piece.Pawn, imba_chess_native.Piece.Knight,
            imba_chess_native.Piece.Bishop, imba_chess_native.Piece.Rook,
            imba_chess_native.Piece.Queen, imba_chess_native.Piece.King,
        ]
        assert len(pieces) == imba_chess_native.Piece.NUM
        assert len(set(pieces)) == 6

    def test_all(self):
        assert len(imba_chess_native.Piece.ALL) == 6


class TestFile:
    def test_all_files(self):
        assert len(imba_chess_native.File.ALL) == imba_chess_native.File.NUM == 8

    def test_index(self):
        for i, f in enumerate(imba_chess_native.File.ALL):
            assert int(f) == i


class TestRank:
    def test_all_ranks(self):
        assert len(imba_chess_native.Rank.ALL) == imba_chess_native.Rank.NUM == 8


class TestSquare:
    def test_num(self):
        assert imba_chess_native.Square.NUM == 64
        assert len(imba_chess_native.Square.ALL) == 64

    def test_new(self):
        sq = imba_chess_native.Square.new(imba_chess_native.File.A, imba_chess_native.Rank.First)
        assert sq == imba_chess_native.Square.A1

    def test_file_rank(self):
        sq = imba_chess_native.Square.E4
        assert sq.file() == imba_chess_native.File.E
        assert sq.rank() == imba_chess_native.Rank.Fourth

    def test_from_str(self):
        sq = imba_chess_native.Square.from_str("e4")
        assert sq == imba_chess_native.Square.E4

    def test_flip(self):
        assert imba_chess_native.Square.A1.flip_file() == imba_chess_native.Square.H1
        assert imba_chess_native.Square.A1.flip_rank() == imba_chess_native.Square.A8

    def test_offset(self):
        sq = imba_chess_native.Square.A1.offset(1, 2)
        assert sq == imba_chess_native.Square.B3

    def test_try_offset_none(self):
        result = imba_chess_native.Square.A1.try_offset(-1, 0)
        assert result is None

    def test_relative_to(self):
        assert imba_chess_native.Square.A1.relative_to(imba_chess_native.Color.White) == imba_chess_native.Square.A1
        assert imba_chess_native.Square.A1.relative_to(imba_chess_native.Color.Black) == imba_chess_native.Square.A8


class TestGameStatus:
    def test_variants(self):
        assert imba_chess_native.GameStatus.Ongoing != imba_chess_native.GameStatus.Won
        assert imba_chess_native.GameStatus.Won != imba_chess_native.GameStatus.Drawn


# ═══════════════════════════════════════════════════════════════════════════
# BitBoard Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBitBoard:
    def test_empty(self):
        bb = imba_chess_native.BitBoard.EMPTY
        assert len(bb) == 0
        assert bb.is_empty()
        assert not bb

    def test_full(self):
        bb = imba_chess_native.BitBoard.FULL
        assert len(bb) == 64
        assert not bb.is_empty()
        assert bb

    def test_from_square(self):
        bb = imba_chess_native.BitBoard.from_square(imba_chess_native.Square.E4)
        assert len(bb) == 1
        assert bb.has(imba_chess_native.Square.E4)
        assert not bb.has(imba_chess_native.Square.E5)

    def test_bitwise_and(self):
        a = imba_chess_native.BitBoard.from_rank(imba_chess_native.Rank.First)
        b = imba_chess_native.BitBoard.from_file(imba_chess_native.File.A)
        result = a & b
        assert len(result) == 1
        assert result.has(imba_chess_native.Square.A1)

    def test_bitwise_or(self):
        a = imba_chess_native.BitBoard.from_square(imba_chess_native.Square.A1)
        b = imba_chess_native.BitBoard.from_square(imba_chess_native.Square.H8)
        result = a | b
        assert len(result) == 2

    def test_bitwise_xor(self):
        a = imba_chess_native.BitBoard.FULL
        b = imba_chess_native.BitBoard.FULL
        result = a ^ b
        assert result == imba_chess_native.BitBoard.EMPTY

    def test_invert(self):
        assert ~imba_chess_native.BitBoard.EMPTY == imba_chess_native.BitBoard.FULL
        assert ~imba_chess_native.BitBoard.FULL == imba_chess_native.BitBoard.EMPTY

    def test_sub(self):
        full = imba_chess_native.BitBoard.FULL
        corners = imba_chess_native.BitBoard.CORNERS
        result = full - corners
        assert len(result) == 60

    def test_contains(self):
        bb = imba_chess_native.BitBoard.from_square(imba_chess_native.Square.E4)
        assert imba_chess_native.Square.E4 in bb
        assert imba_chess_native.Square.A1 not in bb

    def test_iter(self):
        bb = imba_chess_native.BitBoard.CORNERS
        squares = list(bb)
        assert len(squares) == 4
        assert imba_chess_native.Square.A1 in squares
        assert imba_chess_native.Square.H1 in squares
        assert imba_chess_native.Square.A8 in squares
        assert imba_chess_native.Square.H8 in squares

    def test_int(self):
        bb = imba_chess_native.BitBoard.EMPTY
        assert int(bb) == 0
        bb = imba_chess_native.BitBoard.FULL
        assert int(bb) == (1 << 64) - 1

    def test_from_value(self):
        bb = imba_chess_native.BitBoard(0)
        assert bb == imba_chess_native.BitBoard.EMPTY

    def test_flip_ranks(self):
        bb = imba_chess_native.BitBoard.from_rank(imba_chess_native.Rank.First)
        flipped = bb.flip_ranks()
        assert flipped == imba_chess_native.BitBoard.from_rank(imba_chess_native.Rank.Eighth)

    def test_is_subset(self):
        corners = imba_chess_native.BitBoard.CORNERS
        full = imba_chess_native.BitBoard.FULL
        assert corners.is_subset(full)
        assert not full.is_subset(corners)

    def test_is_superset(self):
        corners = imba_chess_native.BitBoard.CORNERS
        full = imba_chess_native.BitBoard.FULL
        assert full.is_superset(corners)

    def test_next_square(self):
        bb = imba_chess_native.BitBoard.from_square(imba_chess_native.Square.E4)
        assert bb.next_square() == imba_chess_native.Square.E4
        assert imba_chess_native.BitBoard.EMPTY.next_square() is None


# ═══════════════════════════════════════════════════════════════════════════
# Move Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMove:
    def test_create(self):
        mv = imba_chess_native.Move(imba_chess_native.Square.E2, imba_chess_native.Square.E4)
        assert mv.from_square == imba_chess_native.Square.E2
        assert mv.to_square == imba_chess_native.Square.E4
        assert mv.promotion is None

    def test_promotion(self):
        mv = imba_chess_native.Move(
            imba_chess_native.Square.E7, imba_chess_native.Square.E8,
            imba_chess_native.Piece.Queen,
        )
        assert mv.promotion == imba_chess_native.Piece.Queen

    def test_from_str(self):
        mv = imba_chess_native.Move.from_str("e2e4")
        assert str(mv) == "e2e4"
        assert mv.from_square == imba_chess_native.Square.E2
        assert mv.to_square == imba_chess_native.Square.E4

    def test_str_promotion(self):
        mv = imba_chess_native.Move.from_str("e7e8q")
        assert str(mv) == "e7e8q"
        assert mv.promotion == imba_chess_native.Piece.Queen

    def test_equality(self):
        a = imba_chess_native.Move.from_str("e2e4")
        b = imba_chess_native.Move(imba_chess_native.Square.E2, imba_chess_native.Square.E4)
        assert a == b

    def test_hash(self):
        mv = imba_chess_native.Move.from_str("e2e4")
        d = {mv: True}
        assert d[imba_chess_native.Move.from_str("e2e4")]


# ═══════════════════════════════════════════════════════════════════════════
# Board Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBoard:
    def test_default(self):
        board = imba_chess_native.Board()
        assert board.side_to_move() == imba_chess_native.Color.White
        assert board.fullmove_number == 1
        assert board.halfmove_clock == 0

    def test_from_fen(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = imba_chess_native.Board.from_fen(fen)
        assert board == imba_chess_native.Board()

    def test_fen_roundtrip(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        board = imba_chess_native.Board.from_fen(fen)
        assert board.fen() == fen

    def test_startpos(self):
        assert imba_chess_native.Board.startpos() == imba_chess_native.Board()

    def test_pieces(self):
        board = imba_chess_native.Board()
        pawns = board.pieces(imba_chess_native.Piece.Pawn)
        assert len(pawns) == 16

    def test_colors(self):
        board = imba_chess_native.Board()
        white = board.colors(imba_chess_native.Color.White)
        assert len(white) == 16

    def test_colored_pieces(self):
        board = imba_chess_native.Board()
        white_pawns = board.colored_pieces(imba_chess_native.Color.White, imba_chess_native.Piece.Pawn)
        assert len(white_pawns) == 8

    def test_occupied(self):
        board = imba_chess_native.Board()
        assert len(board.occupied()) == 32

    def test_piece_on(self):
        board = imba_chess_native.Board()
        assert board.piece_on(imba_chess_native.Square.E1) == imba_chess_native.Piece.King
        assert board.piece_on(imba_chess_native.Square.E4) is None

    def test_color_on(self):
        board = imba_chess_native.Board()
        assert board.color_on(imba_chess_native.Square.E1) == imba_chess_native.Color.White
        assert board.color_on(imba_chess_native.Square.E8) == imba_chess_native.Color.Black

    def test_king(self):
        board = imba_chess_native.Board()
        assert board.king(imba_chess_native.Color.White) == imba_chess_native.Square.E1
        assert board.king(imba_chess_native.Color.Black) == imba_chess_native.Square.E8

    def test_castle_rights(self):
        board = imba_chess_native.Board()
        rights = board.castle_rights(imba_chess_native.Color.White)
        assert rights.short == imba_chess_native.File.H
        assert rights.long == imba_chess_native.File.A

    def test_generate_moves_startpos(self):
        board = imba_chess_native.Board()
        moves = board.generate_moves()
        assert len(moves) == 20

    def test_play_move(self):
        board = imba_chess_native.Board()
        mv = imba_chess_native.Move.from_str("e2e4")
        board.play(mv)
        assert board.side_to_move() == imba_chess_native.Color.Black
        assert board.piece_on(imba_chess_native.Square.E4) == imba_chess_native.Piece.Pawn
        assert board.piece_on(imba_chess_native.Square.E2) is None

    def test_try_play(self):
        board = imba_chess_native.Board()
        mv = imba_chess_native.Move.from_str("e2e4")
        assert board.try_play(mv) is True
        # Board has been modified by try_play
        assert board.side_to_move() == imba_chess_native.Color.Black

    def test_illegal_move(self):
        board = imba_chess_native.Board()
        mv = imba_chess_native.Move.from_str("e1e8")
        with pytest.raises(ValueError):
            board.play(mv)

    def test_is_legal(self):
        board = imba_chess_native.Board()
        assert board.is_legal(imba_chess_native.Move.from_str("e2e4"))
        assert not board.is_legal(imba_chess_native.Move.from_str("e1e8"))

    def test_checkmate(self):
        """Scholar's mate."""
        board = imba_chess_native.Board()
        moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
        for m in moves:
            board.play(imba_chess_native.Move.from_str(m))
        assert board.status() == imba_chess_native.GameStatus.Won
        # Loser is the side to move
        assert board.side_to_move() == imba_chess_native.Color.Black

    def test_status_ongoing(self):
        board = imba_chess_native.Board()
        assert board.status() == imba_chess_native.GameStatus.Ongoing

    def test_en_passant(self):
        board = imba_chess_native.Board()
        assert board.en_passant() is None
        board.play(imba_chess_native.Move.from_str("e2e4"))
        assert board.en_passant() == imba_chess_native.File.E

    def test_hash(self):
        a = imba_chess_native.Board()
        b = imba_chess_native.Board()
        assert a.hash() == b.hash()

    def test_null_move(self):
        board = imba_chess_native.Board()
        nm = board.null_move()
        assert nm is not None
        assert nm.side_to_move() == imba_chess_native.Color.Black

    def test_same_position(self):
        a = imba_chess_native.Board()
        b = imba_chess_native.Board()
        assert a.same_position(b)

    def test_generate_moves_for(self):
        board = imba_chess_native.Board()
        knights = board.pieces(imba_chess_native.Piece.Knight)
        knight_moves = board.generate_moves_for(knights)
        assert len(knight_moves) == 4

    def test_kiwipete_moves(self):
        fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        board = imba_chess_native.Board.from_fen(fen)
        moves = board.generate_moves()
        assert len(moves) == 48

    def test_copy(self):
        board = imba_chess_native.Board()
        board2 = copy.copy(board)
        board.play(imba_chess_native.Move.from_str("e2e4"))
        assert board != board2  # copy is independent

    def test_deepcopy(self):
        board = imba_chess_native.Board()
        board2 = copy.deepcopy(board)
        board.play(imba_chess_native.Move.from_str("e2e4"))
        assert board != board2

    def test_chess960(self):
        board = imba_chess_native.Board.chess960_startpos(518)
        assert board == imba_chess_native.Board()

    def test_pinned(self):
        board = imba_chess_native.Board()
        pinned = board.pinned()
        assert pinned.is_empty()

    def test_checkers(self):
        board = imba_chess_native.Board()
        assert board.checkers().is_empty()

    def test_str_returns_fen(self):
        board = imba_chess_native.Board()
        expected = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert str(board) == expected

    def test_pretty(self):
        board = imba_chess_native.Board()
        pretty = board.pretty()
        assert "K" in pretty  # White king
        assert "k" in pretty  # Black king

    def test_set_halfmove_clock(self):
        board = imba_chess_native.Board()
        board.set_halfmove_clock(50)
        assert board.halfmove_clock == 50

    def test_set_fullmove_number(self):
        board = imba_chess_native.Board()
        board.set_fullmove_number(10)
        assert board.fullmove_number == 10

    def test_generate_piece_moves(self):
        board = imba_chess_native.Board()
        piece_moves = board.generate_piece_moves()
        total = sum(len(pm) for pm in piece_moves)
        assert total == 20


# ═══════════════════════════════════════════════════════════════════════════
# PieceMoves Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPieceMoves:
    def test_iteration(self):
        board = imba_chess_native.Board()
        piece_moves_list = board.generate_piece_moves()
        for pm in piece_moves_list:
            assert len(pm) > 0
            for mv in pm:
                assert isinstance(mv, imba_chess_native.Move)

    def test_properties(self):
        board = imba_chess_native.Board()
        piece_moves_list = board.generate_piece_moves()
        for pm in piece_moves_list:
            assert isinstance(pm.piece, imba_chess_native.Piece)
            assert isinstance(pm.from_square, imba_chess_native.Square)
            assert isinstance(pm.to, imba_chess_native.BitBoard)


# ═══════════════════════════════════════════════════════════════════════════
# CastleRights Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCastleRights:
    def test_default_rights(self):
        board = imba_chess_native.Board()
        rights = board.castle_rights(imba_chess_native.Color.White)
        assert rights.short == imba_chess_native.File.H
        assert rights.long == imba_chess_native.File.A
        assert rights.has_short()
        assert rights.has_long()

    def test_lost_rights(self):
        board = imba_chess_native.Board()
        board.play(imba_chess_native.Move.from_str("e2e4"))
        board.play(imba_chess_native.Move.from_str("e7e5"))
        board.play(imba_chess_native.Move.from_str("e1e2"))  # King moves, loses rights
        rights = board.castle_rights(imba_chess_native.Color.White)
        assert rights.short is None
        assert rights.long is None


# ═══════════════════════════════════════════════════════════════════════════
# BoardBuilder Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBoardBuilder:
    def test_default_builds_startpos(self):
        builder = imba_chess_native.BoardBuilder()
        board = builder.build()
        assert board == imba_chess_native.Board()

    def test_from_board(self):
        board = imba_chess_native.Board()
        builder = imba_chess_native.BoardBuilder.from_board(board)
        rebuilt = builder.build()
        assert rebuilt == board

    def test_empty(self):
        builder = imba_chess_native.BoardBuilder.empty()
        assert builder.piece_on(imba_chess_native.Square.A1) is None

    def test_set_and_clear_piece(self):
        builder = imba_chess_native.BoardBuilder.empty()
        builder.set_piece(imba_chess_native.Square.E1, imba_chess_native.Piece.King, imba_chess_native.Color.White)
        assert builder.piece_on(imba_chess_native.Square.E1) == imba_chess_native.Piece.King
        assert builder.color_on(imba_chess_native.Square.E1) == imba_chess_native.Color.White
        builder.clear_piece(imba_chess_native.Square.E1)
        assert builder.piece_on(imba_chess_native.Square.E1) is None

    def test_side_to_move(self):
        builder = imba_chess_native.BoardBuilder()
        assert builder.side_to_move == imba_chess_native.Color.White
        builder.set_side_to_move(imba_chess_native.Color.Black)
        assert builder.side_to_move == imba_chess_native.Color.Black


# ═══════════════════════════════════════════════════════════════════════════
# Free Functions Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFunctions:
    def test_king_moves(self):
        moves = imba_chess_native.get_king_moves(imba_chess_native.Square.E4)
        assert len(moves) == 8

    def test_king_moves_corner(self):
        moves = imba_chess_native.get_king_moves(imba_chess_native.Square.A1)
        assert len(moves) == 3

    def test_knight_moves(self):
        moves = imba_chess_native.get_knight_moves(imba_chess_native.Square.E4)
        assert len(moves) == 8

    def test_knight_moves_corner(self):
        moves = imba_chess_native.get_knight_moves(imba_chess_native.Square.A1)
        assert len(moves) == 2

    def test_rook_moves_empty(self):
        moves = imba_chess_native.get_rook_moves(imba_chess_native.Square.E4, imba_chess_native.BitBoard.EMPTY)
        assert len(moves) == 14

    def test_bishop_moves_empty(self):
        moves = imba_chess_native.get_bishop_moves(imba_chess_native.Square.E4, imba_chess_native.BitBoard.EMPTY)
        assert len(moves) == 13

    def test_rook_rays(self):
        rays = imba_chess_native.get_rook_rays(imba_chess_native.Square.E4)
        assert len(rays) == 14

    def test_bishop_rays(self):
        rays = imba_chess_native.get_bishop_rays(imba_chess_native.Square.E4)
        assert len(rays) == 13

    def test_pawn_attacks(self):
        attacks = imba_chess_native.get_pawn_attacks(imba_chess_native.Square.E4, imba_chess_native.Color.White)
        assert len(attacks) == 2
        assert attacks.has(imba_chess_native.Square.D5)
        assert attacks.has(imba_chess_native.Square.F5)

    def test_pawn_quiets(self):
        # From starting position, E2 pawn can move to E3 and E4
        quiets = imba_chess_native.get_pawn_quiets(
            imba_chess_native.Square.E2,
            imba_chess_native.Color.White,
            imba_chess_native.BitBoard.EMPTY,
        )
        assert len(quiets) == 2

    def test_between_rays(self):
        between = imba_chess_native.get_between_rays(imba_chess_native.Square.A1, imba_chess_native.Square.H8)
        # Should have the diagonal squares between
        assert len(between) > 0

    def test_line_rays(self):
        line = imba_chess_native.get_line_rays(imba_chess_native.Square.A1, imba_chess_native.Square.H8)
        assert len(line) > 0


# ═══════════════════════════════════════════════════════════════════════════
# MoveProjector Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMoveProjector:
    def test_returns_aligned_canonical_lists(self):
        ucis = [
            "a2a3", "a2a4", "b1a3", "b1c3", "b2b3", "b2b4",
            "c2c3", "c2c4", "d2d3", "d2d4", "e2e3", "e2e4",
            "f2f3", "f2f4", "g1f3", "g1h3", "g2g3", "g2g4",
            "h2h3", "h2h4",
        ]
        # Ids deliberately anti-correlated with UCI order: a mapping-order or
        # id-order sort would pass with sorted ids, so make it fail loudly.
        mapping = {uci: 1000 - index for index, uci in enumerate(reversed(ucis))}
        projector = imba_chess_native.MoveProjector(mapping)

        ids, moves, got_ucis, total = projector.project(imba_chess_native.Board())

        assert total == 20
        assert got_ucis == sorted(ucis)
        assert ids == [mapping[uci] for uci in got_ucis]
        assert [str(move) for move in moves] == got_ucis

    def test_drops_unmapped_moves_but_keeps_the_total(self):
        projector = imba_chess_native.MoveProjector({"e2e4": 5, "d2d4": 3})

        ids, moves, ucis, total = projector.project(imba_chess_native.Board())

        assert total == 20
        assert ucis == ["d2d4", "e2e4"]
        assert ids == [3, 5]
        assert [str(move) for move in moves] == ["d2d4", "e2e4"]

    def test_returns_empty_lists_when_nothing_maps(self):
        projector = imba_chess_native.MoveProjector({"a7a6": 1})

        ids, moves, ucis, total = projector.project(imba_chess_native.Board())

        assert (ids, moves, ucis) == ([], [], [])
        assert total == 20

    def test_normalizes_castling_but_returns_playable_raw_moves(self):
        board = imba_chess_native.Board.from_fen(
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        )
        projector = imba_chess_native.MoveProjector({"e1c1": 7, "e1g1": 9})

        ids, moves, ucis, total = projector.project(board)

        assert total > 2
        assert ids == [7, 9]
        assert ucis == ["e1c1", "e1g1"]
        assert [str(move) for move in moves] == ["e1a1", "e1h1"]
        for move in moves:
            copy_board = board.__copy__()
            copy_board.play(move)

    def test_does_not_normalize_a_non_king_move_onto_a_castle_square(self):
        # A queen on e1 reaching a1 stringifies as "e1a1", which is also the
        # raw form of white long castling. Only the king's move normalizes.
        board = imba_chess_native.Board.from_fen("k7/8/8/8/8/7K/8/4Q3 w - - 0 1")
        # Map only the raw form: if normalization wrongly fired, "e1a1" would
        # become "e1c1" and the move would drop out entirely.
        projector = imba_chess_native.MoveProjector({"e1a1": 1})

        ids, moves, ucis, total = projector.project(board)

        assert total > 2
        assert ids == [1]
        assert ucis == ["e1a1"]
        assert [str(move) for move in moves] == ["e1a1"]

    def test_keeps_vocabularies_isolated(self):
        board = imba_chess_native.Board()
        first = imba_chess_native.MoveProjector({"e2e4": 11})
        second = imba_chess_native.MoveProjector({"e2e4": 29})

        assert first.project(board)[0] == [11]
        assert second.project(board)[0] == [29]

    def test_rejects_non_uci_keys(self):
        with pytest.raises(ValueError, match="invalid UCI move"):
            imba_chess_native.MoveProjector({"<pad>": 0})

    def test_rejects_a_chess960_position_where_one_token_covers_two_moves(self):
        # King on b1 with a rook on a1: the long castle normalizes to "b1c1",
        # and the ordinary king step b1->c1 already IS "b1c1". Two distinct
        # legal moves, one token. Returning both would give the caller a
        # duplicate id/UCI pair with no way to tell them apart, and any dict
        # keyed by UCI would silently drop one -- so refuse loudly instead.
        board = imba_chess_native.Board.from_fen(
            "4k3/8/8/8/8/8/8/RK6 w A - 0 1", shredder=True
        )
        projector = imba_chess_native.MoveProjector({"b1c1": 5})

        with pytest.raises(ValueError, match="ambiguous vocabulary token"):
            projector.project(board)

    def test_allows_the_collision_position_when_the_token_is_unmapped(self):
        # Only mapped moves can collide: an unmapped one is dropped before the
        # check, so the same board projects fine against a vocabulary that
        # simply does not contain the ambiguous token.
        board = imba_chess_native.Board.from_fen(
            "4k3/8/8/8/8/8/8/RK6 w A - 0 1", shredder=True
        )
        projector = imba_chess_native.MoveProjector({"a1a5": 3})

        ids, _moves, ucis, total = projector.project(board)

        assert ids == [3]
        assert ucis == ["a1a5"]
        assert total > 2

    def test_normalizes_chess960_castling_from_the_board(self):
        # Rooks on b1/h1, king on e1. The raw long castle is "e1b1", which is
        # absent from the Python fast path's four-entry castle table, so the
        # destination file has to come from the board. Deliberately not the
        # king-on-b1 arrangement: there "b1c1" is both a normal king step and
        # the normalized long castle, and one token covers two legal moves.
        board = imba_chess_native.Board.from_fen(
            "4k3/8/8/8/8/8/8/1R2K2R w BH - 0 1",
            shredder=True,
        )
        projector = imba_chess_native.MoveProjector({"e1c1": 31, "e1g1": 37})

        ids, moves, ucis, total = projector.project(board)

        assert total > 2
        assert ids == [31, 37]
        assert ucis == ["e1c1", "e1g1"]
        assert [str(move) for move in moves] == ["e1b1", "e1h1"]
        for move in moves:
            copy_board = board.__copy__()
            copy_board.play(move)

    def test_maps_promotions_independently(self):
        board = imba_chess_native.Board.from_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        projector = imba_chess_native.MoveProjector({"a7a8q": 4, "a7a8n": 2})

        ids, moves, ucis, total = projector.project(board)

        assert total > 2
        assert ucis == ["a7a8n", "a7a8q"]
        assert ids == [2, 4]
        assert [str(move) for move in moves] == ucis
