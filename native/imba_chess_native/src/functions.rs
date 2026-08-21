use pyo3::prelude::*;

use crate::bitboard::BitBoard;
use crate::enums::{Color, Square};

// ── Free functions ─────────────────────────────────────────────────────────

#[pyfunction]
fn get_bishop_moves(square: &Square, blockers: &BitBoard) -> BitBoard {
    BitBoard(cozy_chess::get_bishop_moves(square.0, blockers.0))
}

#[pyfunction]
fn get_rook_moves(square: &Square, blockers: &BitBoard) -> BitBoard {
    BitBoard(cozy_chess::get_rook_moves(square.0, blockers.0))
}

#[pyfunction]
fn get_bishop_rays(square: &Square) -> BitBoard {
    BitBoard(cozy_chess::get_bishop_rays(square.0))
}

#[pyfunction]
fn get_rook_rays(square: &Square) -> BitBoard {
    BitBoard(cozy_chess::get_rook_rays(square.0))
}

#[pyfunction]
fn get_king_moves(square: &Square) -> BitBoard {
    BitBoard(cozy_chess::get_king_moves(square.0))
}

#[pyfunction]
fn get_knight_moves(square: &Square) -> BitBoard {
    BitBoard(cozy_chess::get_knight_moves(square.0))
}

#[pyfunction]
fn get_pawn_attacks(square: &Square, color: &Color) -> BitBoard {
    BitBoard(cozy_chess::get_pawn_attacks(square.0, color.0))
}

#[pyfunction]
fn get_pawn_quiets(square: &Square, color: &Color, blockers: &BitBoard) -> BitBoard {
    BitBoard(cozy_chess::get_pawn_quiets(square.0, color.0, blockers.0))
}

#[pyfunction]
fn get_between_rays(square_a: &Square, square_b: &Square) -> BitBoard {
    BitBoard(cozy_chess::get_between_rays(square_a.0, square_b.0))
}

#[pyfunction]
fn get_line_rays(square_a: &Square, square_b: &Square) -> BitBoard {
    BitBoard(cozy_chess::get_line_rays(square_a.0, square_b.0))
}

// ── Register ───────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_bishop_moves, m)?)?;
    m.add_function(wrap_pyfunction!(get_rook_moves, m)?)?;
    m.add_function(wrap_pyfunction!(get_bishop_rays, m)?)?;
    m.add_function(wrap_pyfunction!(get_rook_rays, m)?)?;
    m.add_function(wrap_pyfunction!(get_king_moves, m)?)?;
    m.add_function(wrap_pyfunction!(get_knight_moves, m)?)?;
    m.add_function(wrap_pyfunction!(get_pawn_attacks, m)?)?;
    m.add_function(wrap_pyfunction!(get_pawn_quiets, m)?)?;
    m.add_function(wrap_pyfunction!(get_between_rays, m)?)?;
    m.add_function(wrap_pyfunction!(get_line_rays, m)?)?;
    Ok(())
}
