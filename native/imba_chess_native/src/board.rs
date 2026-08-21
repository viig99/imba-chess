use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::bitboard::BitBoard;
use crate::castle_rights::CastleRights;
use crate::chess_move::ChessMove;
use crate::enums::{Color, File, GameStatus, Piece, Square};
use crate::piece_moves::PieceMoves;

// ── Board ──────────────────────────────────────────────────────────────────

#[pyclass(eq)]
#[derive(Clone, PartialEq, Eq)]
pub struct Board(pub cozy_chess::Board);

#[pymethods]
impl Board {
    // ── Constructors ───────────────────────────────────────────────────

    #[new]
    fn py_new() -> Self {
        Board(cozy_chess::Board::default())
    }

    #[staticmethod]
    fn startpos() -> Self {
        Board(cozy_chess::Board::startpos())
    }

    #[staticmethod]
    #[pyo3(signature = (fen, shredder=false))]
    fn from_fen(fen: &str, shredder: bool) -> PyResult<Self> {
        cozy_chess::Board::from_fen(fen, shredder)
            .map(Board)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        s.parse::<cozy_chess::Board>()
            .map(Board)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    #[staticmethod]
    fn chess960_startpos(scharnagl_number: u32) -> PyResult<Self> {
        if scharnagl_number >= 960 {
            return Err(PyValueError::new_err(
                "Scharnagl number must be in range 0..960",
            ));
        }
        Ok(Board(cozy_chess::Board::chess960_startpos(
            scharnagl_number,
        )))
    }

    #[staticmethod]
    fn double_chess960_startpos(white_scharnagl: u32, black_scharnagl: u32) -> PyResult<Self> {
        if white_scharnagl >= 960 || black_scharnagl >= 960 {
            return Err(PyValueError::new_err(
                "Scharnagl numbers must be in range 0..960",
            ));
        }
        Ok(Board(cozy_chess::Board::double_chess960_startpos(
            white_scharnagl,
            black_scharnagl,
        )))
    }

    // ── Query ──────────────────────────────────────────────────────────

    fn pieces(&self, piece: &Piece) -> BitBoard {
        BitBoard(self.0.pieces(piece.0))
    }

    fn colors(&self, color: &Color) -> BitBoard {
        BitBoard(self.0.colors(color.0))
    }

    fn colored_pieces(&self, color: &Color, piece: &Piece) -> BitBoard {
        BitBoard(self.0.colored_pieces(color.0, piece.0))
    }

    fn occupied(&self) -> BitBoard {
        BitBoard(self.0.occupied())
    }

    fn side_to_move(&self) -> Color {
        Color(self.0.side_to_move())
    }

    fn castle_rights(&self, color: &Color) -> CastleRights {
        CastleRights(*self.0.castle_rights(color.0))
    }

    fn en_passant(&self) -> Option<File> {
        self.0.en_passant().map(File)
    }

    fn hash(&self) -> u64 {
        self.0.hash()
    }

    fn hash_without_ep(&self) -> u64 {
        self.0.hash_without_ep()
    }

    fn pinned(&self) -> BitBoard {
        BitBoard(self.0.pinned())
    }

    fn checkers(&self) -> BitBoard {
        BitBoard(self.0.checkers())
    }

    #[getter]
    fn halfmove_clock(&self) -> u8 {
        self.0.halfmove_clock()
    }

    fn set_halfmove_clock(&mut self, n: u8) -> PyResult<()> {
        if n > 100 {
            return Err(PyValueError::new_err("Halfmove clock must be <= 100"));
        }
        self.0.set_halfmove_clock(n);
        Ok(())
    }

    #[getter]
    fn fullmove_number(&self) -> u16 {
        self.0.fullmove_number()
    }

    fn set_fullmove_number(&mut self, n: u16) -> PyResult<()> {
        if n == 0 {
            return Err(PyValueError::new_err("Fullmove number must be > 0"));
        }
        self.0.set_fullmove_number(n);
        Ok(())
    }

    fn piece_on(&self, square: &Square) -> Option<Piece> {
        self.0.piece_on(square.0).map(Piece)
    }

    fn color_on(&self, square: &Square) -> Option<Color> {
        self.0.color_on(square.0).map(Color)
    }

    fn king(&self, color: &Color) -> Square {
        Square(self.0.king(color.0))
    }

    fn status(&self) -> GameStatus {
        GameStatus(self.0.status())
    }

    fn same_position(&self, other: &Self) -> bool {
        self.0.same_position(&other.0)
    }

    fn is_legal(&self, mv: &ChessMove) -> bool {
        self.0.is_legal(mv.0)
    }

    // ── Gameplay ────────────────────────────────────────────────────────

    fn play(&mut self, mv: &ChessMove) -> PyResult<()> {
        self.0
            .try_play(mv.0)
            .map_err(|e| PyValueError::new_err(format!("Illegal move: {}", e)))
    }

    fn try_play(&mut self, mv: &ChessMove) -> bool {
        self.0.try_play(mv.0).is_ok()
    }

    fn play_unchecked(&mut self, mv: &ChessMove) {
        self.0.play_unchecked(mv.0);
    }

    fn null_move(&self) -> Option<Self> {
        self.0.null_move().map(Board)
    }

    // ── Move generation ────────────────────────────────────────────────

    fn generate_moves(&self) -> Vec<ChessMove> {
        let mut moves = Vec::new();
        self.0.generate_moves(|piece_moves| {
            moves.extend(piece_moves.into_iter().map(ChessMove));
            false
        });
        moves
    }

    fn generate_moves_for(&self, mask: &BitBoard) -> Vec<ChessMove> {
        let mut moves = Vec::new();
        self.0.generate_moves_for(mask.0, |piece_moves| {
            moves.extend(piece_moves.into_iter().map(ChessMove));
            false
        });
        moves
    }

    fn generate_piece_moves(&self) -> Vec<PieceMoves> {
        let mut result = Vec::new();
        self.0.generate_moves(|pm| {
            result.push(PieceMoves(pm));
            false
        });
        result
    }

    fn generate_piece_moves_for(&self, mask: &BitBoard) -> Vec<PieceMoves> {
        let mut result = Vec::new();
        self.0.generate_moves_for(mask.0, |pm| {
            result.push(PieceMoves(pm));
            false
        });
        result
    }

    // ── FEN ────────────────────────────────────────────────────────────

    fn fen(&self) -> String {
        format!("{}", self.0)
    }

    fn shredder_fen(&self) -> String {
        format!("{:#}", self.0)
    }

    // ── Python protocols ───────────────────────────────────────────────

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
    fn __repr__(&self) -> String {
        format!("Board(\"{}\")", self.0)
    }

    fn __hash__(&self) -> u64 {
        self.0.hash()
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }
    fn __deepcopy__(&self, _memo: &Bound<'_, pyo3::types::PyAny>) -> Self {
        self.clone()
    }

    fn pretty(&self) -> String {
        let mut s = String::new();
        s.push_str("  a b c d e f g h\n");
        for rank_idx in (0u8..8).rev() {
            s.push_str(&format!("{} ", rank_idx + 1));
            for file_idx in 0u8..8 {
                let sq = cozy_chess::Square::new(
                    cozy_chess::File::index(file_idx as usize),
                    cozy_chess::Rank::index(rank_idx as usize),
                );
                let ch = match (self.0.color_on(sq), self.0.piece_on(sq)) {
                    (Some(cozy_chess::Color::White), Some(p)) => match p {
                        cozy_chess::Piece::Pawn => 'P',
                        cozy_chess::Piece::Knight => 'N',
                        cozy_chess::Piece::Bishop => 'B',
                        cozy_chess::Piece::Rook => 'R',
                        cozy_chess::Piece::Queen => 'Q',
                        cozy_chess::Piece::King => 'K',
                    },
                    (Some(cozy_chess::Color::Black), Some(p)) => match p {
                        cozy_chess::Piece::Pawn => 'p',
                        cozy_chess::Piece::Knight => 'n',
                        cozy_chess::Piece::Bishop => 'b',
                        cozy_chess::Piece::Rook => 'r',
                        cozy_chess::Piece::Queen => 'q',
                        cozy_chess::Piece::King => 'k',
                    },
                    _ => '.',
                };
                s.push(ch);
                if file_idx < 7 {
                    s.push(' ');
                }
            }
            s.push_str(&format!(" {}\n", rank_idx + 1));
        }
        s.push_str("  a b c d e f g h");
        s
    }
}

// ── Register ───────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Board>()?;
    Ok(())
}
