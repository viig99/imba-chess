use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::board::Board;
use crate::chess_move::ChessMove;
use crate::leaf_eval::LegalSet;

// ── MoveProjector ──────────────────────────────────────────────────────────

/// (vocab ids, raw moves, UCI strings, forcing flags, total legal count) --
/// the first four index-aligned.
type Projection = (
    Vec<i64>,
    Vec<ChessMove>,
    Vec<Py<PyString>>,
    Vec<bool>,
    usize,
);

/// One vocabulary entry: the id to emit, the UCI text projection sorts by, and
/// a reusable Python string so a hit never allocates one.
struct ProjectionEntry {
    id: i64,
    sort_key: String,
    // Position of sort_key among all entries' keys: sorting a projection by it
    // orders exactly as comparing the strings would.
    rank: u32,
    py_uci: Py<PyString>,
    // Shared text for native legal sets; cloning an Arc never allocates.
    uci: std::sync::Arc<str>,
}

/// An immutable move vocabulary that projects a board's legal moves onto ids.
///
/// Owns its mapping outright, so two projectors built from different
/// vocabularies never share state.
#[pyclass(frozen)]
pub struct MoveProjector {
    entries: Vec<ProjectionEntry>,
    // Direct index by (from, to, promotion): entry index + 1, or 0 if the
    // move is not in the vocabulary. Replaces a hashed lookup per legal move.
    table: Vec<u32>,
}

/// Table slot of a move: 64 x 64 squares x (no promotion + six piece kinds).
fn slot(mv: cozy_chess::Move) -> usize {
    let promotion = mv.promotion.map_or(0, |piece| piece as usize + 1);
    (mv.from as usize * 64 + mv.to as usize) * 7 + promotion
}

const TABLE_SLOTS: usize = 64 * 64 * 7;

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

/// Does this LEGAL move capture a piece?
///
/// Mirrors the Python `is_capture_cozy` exactly, including its two subtleties:
/// castling is king-takes-own-rook and must not count, and a pawn reaching an
/// empty square diagonally can only be en passant, so legality alone settles
/// it with no ep-square lookup.
fn is_capture(board: &cozy_chess::Board, mv: cozy_chess::Move) -> bool {
    let moving_piece = board.piece_on(mv.from);
    match board.piece_on(mv.to) {
        None => moving_piece == Some(cozy_chess::Piece::Pawn) && mv.to.file() != mv.from.file(),
        Some(_) => {
            !(moving_piece == Some(cozy_chess::Piece::King)
                && board.color_on(mv.to) == Some(board.side_to_move()))
        }
    }
}

/// Does this LEGAL move give check?
///
/// Simulate-and-look, exactly as the Python it replaces did: play the move on
/// a copy and ask for checkers. A from-scratch direct/discovered-check
/// derivation would be faster still but has to get discovered checks, en
/// passant discoveries, castling rook checks and promotion checks all right --
/// and this feeds search labels that are gated on bit-identical output, so
/// correctness by construction wins. The copy is a native struct copy here,
/// not an FFI crossing per move as before.
fn gives_check(board: &cozy_chess::Board, mv: cozy_chess::Move) -> bool {
    let mut after = board.clone();
    after.play_unchecked(mv);
    !after.checkers().is_empty()
}

/// Promotion, capture, or check -- the search's "forcing" predicate.
///
/// Short-circuits in the same order the Python did, so `gives_check` (much the
/// most expensive of the three) is skipped whenever the cheap tests already
/// resolve the move.
pub(crate) fn is_forcing(board: &cozy_chess::Board, mv: cozy_chess::Move) -> bool {
    mv.promotion.is_some() || is_capture(board, mv) || gives_check(board, mv)
}

impl MoveProjector {
    /// Mapped legal moves in canonical UCI order, plus the total legal count.
    fn select(
        &self,
        inner: &cozy_chess::Board,
    ) -> PyResult<(Vec<(&ProjectionEntry, cozy_chess::Move)>, usize)> {
        let mut total = 0usize;
        let mut selected: Vec<(&ProjectionEntry, cozy_chess::Move)> = Vec::with_capacity(32);
        inner.generate_moves(|piece_moves| {
            for mv in piece_moves {
                total += 1;
                let index = self.table[slot(lookup_move(inner, mv))];
                if index != 0 {
                    selected.push((&self.entries[index as usize - 1], mv));
                }
            }
            false
        });
        // Stable, and by UCI string, so the order matches the Python oracle's
        // `sorted(..., key=lambda i: ucis[i])`.
        selected.sort_by_key(|entry| entry.0.rank);

        // Two distinct legal moves can normalize onto the same token when the
        // king does not start on the e-file: a plain king step to c1/g1 lands
        // exactly where the long/short castle normalizes. Standard chess cannot
        // produce this -- a king on e1 can never step to c1 or g1 -- but a
        // Chess960 board with the king on b1 or d1 can. Returning both would
        // hand the caller two entries with an identical id and UCI and no way
        // to tell them apart, and anything building a dict keyed by UCI would
        // silently drop a legal move. Refuse instead: the ambiguity lives in
        // the from-to token space itself, not in something this function can
        // resolve. Two distinct entries can never share a sort_key -- they come
        // from Python dict keys, which are unique -- so a collision is exactly
        // the same entry reached twice, and pointer equality decides it without
        // touching the strings. Colliding entries are adjacent after the sort.
        if let Some(pair) = selected.windows(2).find(|w| std::ptr::eq(w[0].0, w[1].0)) {
            return Err(PyValueError::new_err(format!(
                "ambiguous vocabulary token {:?}: legal moves {} and {} both map \
                 to it. A castle and a plain king step share a destination, which \
                 the from-to token space cannot distinguish. Unreachable in \
                 standard chess; unsupported for this Chess960 castling geometry.",
                pair[0].0.sort_key, pair[0].1, pair[1].1,
            )));
        }
        Ok((selected, total))
    }
}

#[pymethods]
impl MoveProjector {
    #[new]
    fn py_new(move_token_to_id: &Bound<'_, PyDict>) -> PyResult<Self> {
        // Moves map to entries; a move parsed twice keeps its last entry.
        let mut index_of: HashMap<cozy_chess::Move, usize> =
            HashMap::with_capacity(move_token_to_id.len());
        let mut entries: Vec<ProjectionEntry> = Vec::with_capacity(move_token_to_id.len());
        for (key, value) in move_token_to_id.iter() {
            let py_uci = key
                .downcast::<PyString>()
                .map_err(|_| PyValueError::new_err(format!("invalid UCI move: {}", key)))?;
            let uci = py_uci.to_str()?.to_owned();
            let mv = uci
                .parse::<cozy_chess::Move>()
                .map_err(|_| PyValueError::new_err(format!("invalid UCI move: {}", uci)))?;
            let id: i64 = value.extract()?;
            let entry = ProjectionEntry {
                id,
                uci: std::sync::Arc::from(uci.as_str()),
                sort_key: uci,
                rank: 0,
                py_uci: py_uci.clone().unbind(),
            };
            match index_of.get(&mv) {
                Some(&index) => entries[index] = entry,
                None => {
                    index_of.insert(mv, entries.len());
                    entries.push(entry);
                }
            }
        }
        let mut order: Vec<usize> = (0..entries.len()).collect();
        order.sort_by(|&a, &b| entries[a].sort_key.cmp(&entries[b].sort_key));
        for (rank, index) in order.into_iter().enumerate() {
            entries[index].rank = rank as u32;
        }
        let mut table = vec![0u32; TABLE_SLOTS];
        for (mv, index) in index_of {
            table[slot(mv)] = index as u32 + 1;
        }
        Ok(MoveProjector { entries, table })
    }

    /// (ids, moves, UCIs) for the mapped legal moves in canonical UCI order,
    /// plus the total legal move count.
    ///
    /// The four lists stay index-aligned so a caller can gather logits by id
    /// and still know which move each one belongs to, and whether it is
    /// forcing (promotion, capture, or check -- what the search's refutation
    /// floor selects on). Forcing is computed here because this call has
    /// already generated the moves and holds the board: the search used to
    /// re-walk the identical (board, moves) pair afterwards, at ~19.8 us per
    /// node and two FFI crossings per move. Moves come back in their
    /// raw generated form -- normalization is for the vocabulary only, and a
    /// normalized castle is not playable. Unmapped moves are dropped but still
    /// counted in the total.
    ///
    /// Raises ValueError if two legal moves normalize onto one token, which
    /// only Chess960 castling geometry can produce; see the check below.
    fn project(&self, py: Python<'_>, board: &Board) -> PyResult<Projection> {
        let inner = &board.0;
        let (selected, total) = self.select(inner)?;
        let mut ids = Vec::with_capacity(selected.len());
        let mut moves = Vec::with_capacity(selected.len());
        let mut ucis = Vec::with_capacity(selected.len());
        let mut forcing = Vec::with_capacity(selected.len());
        for (entry, mv) in selected {
            ids.push(entry.id);
            moves.push(ChessMove(mv));
            ucis.push(entry.py_uci.clone_ref(py));
            forcing.push(is_forcing(inner, mv));
        }
        Ok((ids, moves, ucis, forcing, total))
    }

    /// The same projection held natively: no per-move Python objects.
    fn project_lean(&self, board: &Board) -> PyResult<LegalSet> {
        let inner = &board.0;
        let (selected, total) = self.select(inner)?;
        let mut legal = LegalSet {
            ids: Vec::with_capacity(selected.len()),
            moves: Vec::with_capacity(selected.len()),
            ucis: Vec::with_capacity(selected.len()),
            // Forcing flags cost a move per legal move; most searches never
            // read them, so they are computed from this board on first use.
            forcing: std::sync::OnceLock::new(),
            board: inner.clone(),
            total,
        };
        for (entry, mv) in selected {
            legal.ids.push(entry.id);
            legal.moves.push(mv);
            legal.ucis.push(entry.uci.clone());
        }
        Ok(legal)
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
