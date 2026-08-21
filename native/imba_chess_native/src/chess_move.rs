use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::enums::{Piece, Square};

// ── Move ───────────────────────────────────────────────────────────────────

#[pyclass(frozen, eq, hash, name = "Move")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChessMove(pub cozy_chess::Move);

// `from_square`/`to_square` are Python attribute names fixed by the wrapper
// API this binding reproduces, not Rust constructors/converters -- renaming
// them to satisfy wrong_self_convention would break every caller.
#[allow(clippy::wrong_self_convention)]
#[pymethods]
impl ChessMove {
    #[new]
    #[pyo3(signature = (from_square, to_square, promotion=None))]
    fn py_new(from_square: &Square, to_square: &Square, promotion: Option<&Piece>) -> Self {
        ChessMove(cozy_chess::Move {
            from: from_square.0,
            to: to_square.0,
            promotion: promotion.map(|p| p.0),
        })
    }

    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        s.parse::<cozy_chess::Move>()
            .map(ChessMove)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    #[getter]
    fn from_square(&self) -> Square {
        Square(self.0.from)
    }

    #[getter]
    fn to_square(&self) -> Square {
        Square(self.0.to)
    }

    #[getter]
    fn promotion(&self) -> Option<Piece> {
        self.0.promotion.map(Piece)
    }

    fn __repr__(&self) -> String {
        format!("Move(\"{}\")", self.0)
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
}

// ── Register ───────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ChessMove>()?;
    Ok(())
}
