use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::board::Board;
use crate::castle_rights::CastleRights;
use crate::enums::{Color, File, Piece, Square};

// ── BoardBuilder ───────────────────────────────────────────────────────────

#[pyclass]
#[derive(Clone)]
pub struct BoardBuilder(pub cozy_chess::BoardBuilder);

#[pymethods]
impl BoardBuilder {
    #[new]
    fn py_new() -> Self {
        BoardBuilder(cozy_chess::BoardBuilder::default())
    }

    #[staticmethod]
    fn empty() -> Self {
        BoardBuilder(cozy_chess::BoardBuilder::empty())
    }

    #[staticmethod]
    fn from_board(board: &Board) -> Self {
        BoardBuilder(cozy_chess::BoardBuilder::from_board(&board.0))
    }

    fn set_piece(&mut self, square: &Square, piece: &Piece, color: &Color) {
        *self.0.square_mut(square.0) = Some((piece.0, color.0));
    }

    fn clear_piece(&mut self, square: &Square) {
        *self.0.square_mut(square.0) = None;
    }

    fn piece_on(&self, square: &Square) -> Option<Piece> {
        self.0.square(square.0).map(|(p, _)| Piece(p))
    }

    fn color_on(&self, square: &Square) -> Option<Color> {
        self.0.square(square.0).map(|(_, c)| Color(c))
    }

    fn square(&self, sq: &Square) -> Option<(Piece, Color)> {
        self.0.square(sq.0).map(|(p, c)| (Piece(p), Color(c)))
    }

    #[getter]
    fn side_to_move(&self) -> Color {
        Color(self.0.side_to_move)
    }

    fn set_side_to_move(&mut self, color: &Color) {
        self.0.side_to_move = color.0;
    }

    fn castle_rights(&self, color: &Color) -> CastleRights {
        CastleRights(*self.0.castle_rights(color.0))
    }

    #[pyo3(signature = (color, short=None, long=None))]
    fn set_castle_rights(&mut self, color: &Color, short: Option<&File>, long: Option<&File>) {
        let rights = self.0.castle_rights_mut(color.0);
        rights.short = short.map(|f| f.0);
        rights.long = long.map(|f| f.0);
    }

    #[getter]
    fn en_passant(&self) -> Option<Square> {
        self.0.en_passant.map(Square)
    }

    fn set_en_passant(&mut self, square: &Square) {
        self.0.en_passant = Some(square.0);
    }

    fn clear_en_passant(&mut self) {
        self.0.en_passant = None;
    }

    #[getter]
    fn halfmove_clock(&self) -> u8 {
        self.0.halfmove_clock
    }

    fn set_halfmove_clock(&mut self, n: u8) {
        self.0.halfmove_clock = n;
    }

    #[getter]
    fn fullmove_number(&self) -> u16 {
        self.0.fullmove_number
    }

    fn set_fullmove_number(&mut self, n: u16) {
        self.0.fullmove_number = n;
    }

    fn build(&self) -> PyResult<Board> {
        self.0
            .build()
            .map(Board)
            .map_err(|e| PyValueError::new_err(format!("Invalid board: {}", e)))
    }

    fn __repr__(&self) -> String {
        format!("BoardBuilder(side_to_move={:?})", self.0.side_to_move)
    }
}

// ── Register ───────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BoardBuilder>()?;
    Ok(())
}
