use pyo3::prelude::*;

use crate::board::Board;
use crate::chess_move::ChessMove;

// ── Terminal detection ─────────────────────────────────────────────────────
//
// Port of cozy_bridge.terminal_value_native and the helpers it drove
// (_no_heavy_pieces, insufficient_material, repetition_hash,
// ep_has_legal_capturer) plus search._cozy_push. Those ran per created child
// -- 3.14 times per evaluated node -- and each was a Python function making
// several FFI crossings, which measured as ~15.5 Board.pieces() calls per node
// evaluation spent on nothing but "is this position over?".
//
// Semantics are transcribed, not reinterpreted: this feeds search labels gated
// on bit-identical rollout output, and tests/test_native_terminal.py keeps the
// Python original as an independent oracle.

const BB_DARK_SQUARES: u64 = 0xAA55_AA55_AA55_AA55;
const BB_LIGHT_SQUARES: u64 = !BB_DARK_SQUARES;

fn no_heavy_pieces(board: &cozy_chess::Board) -> bool {
    let heavy = board.pieces(cozy_chess::Piece::Pawn).0
        | board.pieces(cozy_chess::Piece::Rook).0
        | board.pieces(cozy_chess::Piece::Queen).0;
    heavy == 0
}

/// Exact python-chess `has_insufficient_material(color)` semantics.
///
/// Unlike the Python it replaces, the bitboards are read lazily: the early
/// "we have a pawn/rook/queen" exit is by far the common case, and paying for
/// all eight boards before testing it was most of this function's cost.
fn insufficient_material_for(board: &cozy_chess::Board, color: cozy_chess::Color) -> bool {
    let occ = board.colors(color).0;
    let pawns = board.pieces(cozy_chess::Piece::Pawn).0;
    let rooks = board.pieces(cozy_chess::Piece::Rook).0;
    let queens = board.pieces(cozy_chess::Piece::Queen).0;
    if occ & (pawns | rooks | queens) != 0 {
        return false;
    }

    let knights = board.pieces(cozy_chess::Piece::Knight).0;
    let bishops = board.pieces(cozy_chess::Piece::Bishop).0;
    let kings = board.pieces(cozy_chess::Piece::King).0;
    let other = !color;
    let occ_other = board.colors(other).0;

    // Knights: insufficient only with no other material, and only if the
    // opponent has nothing that would permit a selfmate.
    if occ & knights != 0 {
        return occ.count_ones() <= 2 && (occ_other & !kings & !queens) == 0;
    }

    // Bishops: insufficient only if every bishop on the board -- both
    // colours' -- shares a square colour, and the opponent has no pawns or
    // knights to selfmate with.
    if occ & bishops != 0 {
        let same_color = (bishops & BB_DARK_SQUARES) == 0 || (bishops & BB_LIGHT_SQUARES) == 0;
        return same_color && pawns == 0 && knights == 0;
    }

    true
}

fn insufficient_material(board: &cozy_chess::Board) -> bool {
    insufficient_material_for(board, cozy_chess::Color::White)
        && insufficient_material_for(board, cozy_chess::Color::Black)
}

/// Does the board's en-passant flag have an actual LEGAL capturing pawn move?
///
/// python-chess's transposition key -- what threefold claims key off -- only
/// folds in the ep square when a legal capturer exists, while cozy hashes the
/// flag unconditionally. Without this gate the same position reached via a
/// capturer-less double push would hash differently and repetitions would be
/// undercounted.
fn ep_has_legal_capturer(board: &cozy_chess::Board) -> bool {
    let ep_file = match board.en_passant() {
        None => return false,
        Some(file) => file,
    };
    let stm = board.side_to_move();
    let (from_rank, to_rank) = match stm {
        cozy_chess::Color::White => (cozy_chess::Rank::Fifth, cozy_chess::Rank::Sixth),
        cozy_chess::Color::Black => (cozy_chess::Rank::Fourth, cozy_chess::Rank::Third),
    };
    let pawns = board.colors(stm) & board.pieces(cozy_chess::Piece::Pawn);
    let file_index = ep_file as usize;
    for delta in [-1i32, 1] {
        let adj = file_index as i32 + delta;
        if !(0..8).contains(&adj) {
            continue;
        }
        let from_file = cozy_chess::File::index(adj as usize);
        let from = cozy_chess::Square::new(from_file, from_rank);
        if !pawns.has(from) {
            continue;
        }
        let mv = cozy_chess::Move {
            from,
            to: cozy_chess::Square::new(ep_file, to_rank),
            promotion: None,
        };
        if board.is_legal(mv) {
            return true;
        }
    }
    false
}

fn repetition_hash(board: &cozy_chess::Board) -> u64 {
    if board.en_passant().is_some() && !ep_has_legal_capturer(board) {
        board.hash_without_ep()
    } else {
        board.hash()
    }
}

/// Exact game result from `color`'s POV, or None if the game is not over.
fn terminal_value(
    board: &cozy_chess::Board,
    color_is_stm: bool,
    hash_history: &[u64],
) -> Option<f64> {
    match board.status() {
        // cozy 'Won' means the side to move is checkmated.
        cozy_chess::GameStatus::Won => return Some(if color_is_stm { -1.0 } else { 1.0 }),
        // Covers stalemate AND cozy's own halfmove>=100 auto-draw, which
        // subsumes the plain fifty-move claim.
        cozy_chess::GameStatus::Drawn => return Some(0.0),
        cozy_chess::GameStatus::Ongoing => {}
    }
    if no_heavy_pieces(board) && insufficient_material(board) {
        return Some(0.0);
    }

    let halfmove = board.halfmove_clock() as usize;
    if halfmove < 7 {
        // A third occurrence (or either one-ply-early claim) needs at least
        // seven reversible plies, so the O(history) scan cannot fire.
        return None;
    }

    let current = repetition_hash(board);
    let window = if halfmove < hash_history.len() {
        &hash_history[hash_history.len() - halfmove..]
    } else {
        hash_history
    };
    if window.iter().filter(|h| **h == current).count() >= 2 {
        return Some(0.0); // third occurrence reached
    }

    // python-chess also allows claiming one reversible ply early:
    //  - repetition: any legal move REACHING the third occurrence;
    //  - fifty-move: at halfmove 99, any non-zeroing move whose result still
    //    has a legal move of its own (status() cannot be read for this, since
    //    it reports Drawn at halfmove 100 regardless).
    let claim_fifty_early = halfmove == 99;
    let mut found = false;
    board.generate_moves(|piece_moves| {
        for mv in piece_moves {
            let mut child = board.clone();
            child.play_unchecked(mv);
            if child.halfmove_clock() == 0 {
                continue; // irreversible: cannot repeat, resets the clock
            }
            if claim_fifty_early && child.generate_moves(|_| true) {
                found = true;
                return true;
            }
            let child_hash = repetition_hash(&child);
            let count = window.iter().filter(|h| **h == child_hash).count()
                + usize::from(current == child_hash);
            if count >= 2 {
                found = true;
                return true;
            }
        }
        false
    });
    if found {
        Some(0.0)
    } else {
        None
    }
}

// ── Python surface ─────────────────────────────────────────────────────────

/// (child board, child hash history, terminal value or None).
type PushResult = (Board, Vec<u64>, Option<f64>);

/// Play one tree edge and classify the result in a single crossing.
///
/// The search always did these together -- `_cozy_push` then
/// `terminal_value_native` on its output -- at roughly ten FFI crossings per
/// created child. `child_history` resets on a zeroing move (capture or pawn
/// move) and otherwise carries the parent's history plus the parent's own
/// repetition hash, which is the contract `terminal_value` reads.
#[pyfunction]
pub fn push_and_classify(
    board: &Board,
    mv: &ChessMove,
    hash_history: Vec<u64>,
    color_is_stm: bool,
) -> PushResult {
    let mut child = board.0.clone();
    child.play_unchecked(mv.0);
    let child_history = if child.halfmove_clock() == 0 {
        Vec::new()
    } else {
        let mut history = hash_history;
        history.push(repetition_hash(&board.0));
        history
    };
    let value = terminal_value(&child, color_is_stm, &child_history);
    (Board(child), child_history, value)
}

/// Standalone classification, for boards not reached by pushing an edge.
#[pyfunction]
pub fn terminal_value_of(board: &Board, color_is_stm: bool, hash_history: Vec<u64>) -> Option<f64> {
    terminal_value(&board.0, color_is_stm, &hash_history)
}

/// Canonical repetition/transposition hash, ep-normalized.
#[pyfunction]
pub fn repetition_hash_of(board: &Board) -> u64 {
    repetition_hash(&board.0)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(push_and_classify, m)?)?;
    m.add_function(wrap_pyfunction!(terminal_value_of, m)?)?;
    m.add_function(wrap_pyfunction!(repetition_hash_of, m)?)?;
    Ok(())
}
