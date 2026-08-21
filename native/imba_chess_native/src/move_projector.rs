use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::board::Board;
use crate::chess_move::ChessMove;

// ── MoveProjector ──────────────────────────────────────────────────────────

/// One vocabulary entry: the id to emit, the UCI text projection sorts by, and
/// a reusable Python string so a hit never allocates one.
struct ProjectionEntry {
    id: i64,
    sort_key: String,
    py_uci: Py<PyString>,
}

/// An immutable move vocabulary that projects a board's legal moves onto ids.
///
/// Owns its mapping outright, so two projectors built from different
/// vocabularies never share state.
#[pyclass(frozen)]
pub struct MoveProjector {
    entries: HashMap<cozy_chess::Move, ProjectionEntry>,
}

/// Standard UCI form of a generated move.
///
/// cozy-chess encodes castling as king-takes-own-rook, which stringifies to
/// the rook's square ("e1h1", or "e1b1" on a Chess960 board). Vocabularies
/// speak the king-destination form, so rewrite the destination to the castled
/// king file, taken from the direction of the rook rather than a fixed table.
/// Every other move is already canonical.
fn lookup_move(board: &cozy_chess::Board, mv: cozy_chess::Move) -> cozy_chess::Move {
    let is_castle = board.piece_on(mv.from) == Some(cozy_chess::Piece::King)
        && board.color_on(mv.to) == Some(board.side_to_move());
    if !is_castle {
        return mv;
    }
    let file = if mv.to.file() > mv.from.file() {
        cozy_chess::File::G
    } else {
        cozy_chess::File::C
    };
    cozy_chess::Move {
        from: mv.from,
        to: cozy_chess::Square::new(file, mv.from.rank()),
        promotion: mv.promotion,
    }
}

#[pymethods]
impl MoveProjector {
    #[new]
    fn py_new(move_token_to_id: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut entries = HashMap::with_capacity(move_token_to_id.len());
        for (key, value) in move_token_to_id.iter() {
            let py_uci = key
                .downcast::<PyString>()
                .map_err(|_| PyValueError::new_err(format!("invalid UCI move: {}", key)))?;
            let uci = py_uci.to_str()?.to_owned();
            let mv = uci
                .parse::<cozy_chess::Move>()
                .map_err(|_| PyValueError::new_err(format!("invalid UCI move: {}", uci)))?;
            let id: i64 = value.extract()?;
            entries.insert(
                mv,
                ProjectionEntry {
                    id,
                    sort_key: uci,
                    py_uci: py_uci.clone().unbind(),
                },
            );
        }
        Ok(MoveProjector { entries })
    }

    /// (ids, moves, UCIs) for the mapped legal moves in canonical UCI order,
    /// plus the total legal move count.
    ///
    /// The three lists stay index-aligned so a caller can gather logits by id
    /// and still know which move each one belongs to. Moves come back in their
    /// raw generated form -- normalization is for the vocabulary only, and a
    /// normalized castle is not playable. Unmapped moves are dropped but still
    /// counted in the total.
    fn project(
        &self,
        py: Python<'_>,
        board: &Board,
    ) -> (Vec<i64>, Vec<ChessMove>, Vec<Py<PyString>>, usize) {
        let inner = &board.0;
        let mut total = 0usize;
        let mut selected: Vec<(&ProjectionEntry, cozy_chess::Move)> = Vec::with_capacity(32);
        inner.generate_moves(|piece_moves| {
            for mv in piece_moves {
                total += 1;
                if let Some(entry) = self.entries.get(&lookup_move(inner, mv)) {
                    selected.push((entry, mv));
                }
            }
            false
        });
        // Stable, and by UCI string, so the order matches the Python oracle's
        // `sorted(..., key=lambda i: ucis[i])` even where one token covers two
        // legal moves.
        selected.sort_by(|a, b| a.0.sort_key.cmp(&b.0.sort_key));

        let mut ids = Vec::with_capacity(selected.len());
        let mut moves = Vec::with_capacity(selected.len());
        let mut ucis = Vec::with_capacity(selected.len());
        for (entry, mv) in selected {
            ids.push(entry.id);
            moves.push(ChessMove(mv));
            ucis.push(entry.py_uci.clone_ref(py));
        }
        (ids, moves, ucis, total)
    }

    fn __len__(&self) -> usize {
        self.entries.len()
    }

    fn __repr__(&self) -> String {
        format!("MoveProjector({} moves)", self.entries.len())
    }
}

// ── Register ───────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MoveProjector>()?;
    Ok(())
}
