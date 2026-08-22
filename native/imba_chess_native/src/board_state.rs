use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::board::Board;

// ── Board-state encoding ───────────────────────────────────────────────────
//
// Port of BoardStateEncoder.encode_cozy. It ran once per evaluated search node
// -- 428k times per 20-game rollout -- and made ~10 FFI crossings each time
// (colors, six pieces, two castle_rights, side_to_move, the clocks) before
// bit-twiddling 64 squares in Python. Output is integer ids, so this is exact
// by construction; tests/test_native_board_state.py keeps the Python as the
// oracle.

/// En-passant reporting mode, matching BoardTokenConfig.en_passant.
const EP_MODE_FEN: u8 = 0; // report the flag as-is, FEN style
const EP_MODE_LEGAL: u8 = 1; // only if an actual LEGAL capture exists
const EP_MODE_XFEN: u8 = 2; // only if a pseudo-legal capturer exists

/// White ids 1..6, black 7..12, in Pawn, Knight, Bishop, Rook, Queen, King
/// order -- the same table `_cozy_encode_consts` builds.
const PIECES: [(u8, cozy_chess::Piece); 6] = [
    (1, cozy_chess::Piece::Pawn),
    (2, cozy_chess::Piece::Knight),
    (3, cozy_chess::Piece::Bishop),
    (4, cozy_chess::Piece::Rook),
    (5, cozy_chess::Piece::Queen),
    (6, cozy_chess::Piece::King),
];

fn bucket(value: i64, max_value: i64, bucket_size: i64) -> i64 {
    value.clamp(0, max_value) / bucket_size
}

/// The <=2 pseudo-legal en-passant captures for the board's ep flag.
///
/// A pawn can only move diagonally onto an empty square via en passant, so
/// adjacency plus the right rank is the whole pseudo-legal test.
fn ep_capturers(board: &cozy_chess::Board) -> Vec<cozy_chess::Move> {
    let ep_file = match board.en_passant() {
        None => return Vec::new(),
        Some(file) => file,
    };
    let stm = board.side_to_move();
    let (from_rank, to_rank) = match stm {
        cozy_chess::Color::White => (cozy_chess::Rank::Fifth, cozy_chess::Rank::Sixth),
        cozy_chess::Color::Black => (cozy_chess::Rank::Fourth, cozy_chess::Rank::Third),
    };
    let pawns = board.colors(stm) & board.pieces(cozy_chess::Piece::Pawn);
    let mut moves = Vec::new();
    for delta in [-1i32, 1] {
        let adj = ep_file as i32 + delta;
        if !(0..8).contains(&adj) {
            continue;
        }
        let from = cozy_chess::Square::new(cozy_chess::File::index(adj as usize), from_rank);
        if !pawns.has(from) {
            continue;
        }
        moves.push(cozy_chess::Move {
            from,
            to: cozy_chess::Square::new(ep_file, to_rank),
            promotion: None,
        });
    }
    moves
}

fn ep_file_id(board: &cozy_chess::Board, ep_mode: u8) -> i64 {
    let ep_file = match board.en_passant() {
        None => return 0,
        Some(file) => file,
    };
    let file_idx = ep_file as i64;
    match ep_mode {
        EP_MODE_FEN => file_idx + 1,
        EP_MODE_LEGAL => {
            let capturers = ep_capturers(board);
            if capturers.into_iter().any(|mv| board.is_legal(mv)) {
                file_idx + 1
            } else {
                0
            }
        }
        // xfen: a pseudo-legal capturer is enough; the gap versus "legal" is
        // exactly the ep-pin case, which adjacency alone already admits.
        _ => {
            if ep_capturers(board).is_empty() {
                0
            } else {
                file_idx + 1
            }
        }
    }
}

/// (piece_ids[64], turn_id, castle_id, ep_file_id, halfmove_bucket, fullmove_bucket)
type EncodedState = (Vec<u8>, i64, i64, i64, i64, i64);

/// Encode one board into the model's board-state token ids, in one crossing.
#[pyfunction]
#[pyo3(signature = (
    board, ep_mode, halfmove_max, halfmove_bucket_size, fullmove_max,
    fullmove_bucket_size,
))]
pub fn encode_board_state(
    board: &Board,
    ep_mode: u8,
    halfmove_max: i64,
    halfmove_bucket_size: i64,
    fullmove_max: i64,
    fullmove_bucket_size: i64,
) -> PyResult<EncodedState> {
    if ep_mode > EP_MODE_XFEN {
        return Err(PyValueError::new_err(format!(
            "ep_mode must be 0 (fen), 1 (legal) or 2 (xfen), got {}",
            ep_mode
        )));
    }
    if halfmove_bucket_size <= 0 || fullmove_bucket_size <= 0 {
        return Err(PyValueError::new_err("bucket sizes must be > 0"));
    }
    let inner = &board.0;

    let mut piece_ids = vec![0u8; 64];
    let white = inner.colors(cozy_chess::Color::White);
    for (white_id, piece) in PIECES {
        let bb = inner.pieces(piece);
        for square in bb & white {
            piece_ids[square as usize] = white_id;
        }
        for square in bb & !white {
            piece_ids[square as usize] = white_id + 6;
        }
    }

    let rights_white = inner.castle_rights(cozy_chess::Color::White);
    let rights_black = inner.castle_rights(cozy_chess::Color::Black);
    let castle_id = i64::from(rights_white.short.is_some())
        | (i64::from(rights_white.long.is_some()) << 1)
        | (i64::from(rights_black.short.is_some()) << 2)
        | (i64::from(rights_black.long.is_some()) << 3);

    Ok((
        piece_ids,
        i64::from(inner.side_to_move() == cozy_chess::Color::Black),
        castle_id,
        ep_file_id(inner, ep_mode),
        bucket(
            inner.halfmove_clock() as i64,
            halfmove_max,
            halfmove_bucket_size,
        ),
        bucket(
            inner.fullmove_number() as i64,
            fullmove_max,
            fullmove_bucket_size,
        ),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_board_state, m)?)?;
    Ok(())
}
