use pyo3::prelude::*;

use crate::bitboard::BitBoard;
use crate::chess_move::ChessMove;
use crate::enums::{Piece, Square};

// ── PieceMoves ─────────────────────────────────────────────────────────────

#[pyclass(frozen)]
#[derive(Clone)]
pub struct PieceMoves(pub cozy_chess::PieceMoves);

// `from_square` is a Python attribute name fixed by the wrapper API this
// binding reproduces, not a Rust constructor -- renaming it to satisfy
// wrong_self_convention would break every caller.
#[allow(clippy::wrong_self_convention)]
#[pymethods]
impl PieceMoves {
    #[getter]
    fn piece(&self) -> Piece {
        Piece(self.0.piece)
    }

    #[getter]
    fn from_square(&self) -> Square {
        Square(self.0.from)
    }

    #[getter]
    fn to(&self) -> BitBoard {
        BitBoard(self.0.to)
    }

    fn moves(&self) -> Vec<ChessMove> {
        self.0.into_iter().map(ChessMove).collect()
    }

    fn __iter__(&self) -> PieceMovesIter {
        PieceMovesIter {
            moves: self.0.into_iter().map(ChessMove).collect(),
            index: 0,
        }
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PieceMoves({:?} from {} -> {} targets)",
            self.0.piece,
            self.0.from,
            self.0.to.len()
        )
    }
}

// ── PieceMovesIter ─────────────────────────────────────────────────────────

#[pyclass]
pub struct PieceMovesIter {
    moves: Vec<ChessMove>,
    index: usize,
}

#[pymethods]
impl PieceMovesIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<ChessMove> {
        if self.index < self.moves.len() {
            let m = self.moves[self.index];
            self.index += 1;
            Some(m)
        } else {
            None
        }
    }
}

// ── Register ───────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PieceMoves>()?;
    m.add_class::<PieceMovesIter>()?;
    Ok(())
}
