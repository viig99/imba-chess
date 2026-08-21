#![allow(non_snake_case)]

use pyo3::prelude::*;

pub mod bitboard;
pub mod board;
pub mod board_builder;
pub mod castle_rights;
pub mod chess_move;
pub mod enums;
pub mod functions;
pub mod move_projector;
pub mod piece_moves;
pub mod terminal;

#[pymodule]
fn imba_chess_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    enums::register(m)?;
    bitboard::register(m)?;
    chess_move::register(m)?;
    piece_moves::register(m)?;
    castle_rights::register(m)?;
    board::register(m)?;
    board_builder::register(m)?;
    functions::register(m)?;
    move_projector::register(m)?;
    terminal::register(m)?;
    Ok(())
}
