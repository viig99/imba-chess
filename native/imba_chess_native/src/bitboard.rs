use pyo3::prelude::*;

use crate::enums::Square;

// ── BitBoard ───────────────────────────────────────────────────────────────

#[pyclass(frozen, eq, hash, ord)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BitBoard(pub cozy_chess::BitBoard);

#[pymethods]
impl BitBoard {
    // ── Constants ───────────────────────────────────────────────────────

    #[classattr]
    fn EMPTY() -> Self {
        BitBoard(cozy_chess::BitBoard::EMPTY)
    }
    #[classattr]
    fn FULL() -> Self {
        BitBoard(cozy_chess::BitBoard::FULL)
    }
    #[classattr]
    fn EDGES() -> Self {
        BitBoard(cozy_chess::BitBoard::EDGES)
    }
    #[classattr]
    fn CORNERS() -> Self {
        BitBoard(cozy_chess::BitBoard::CORNERS)
    }
    #[classattr]
    fn DARK_SQUARES() -> Self {
        BitBoard(cozy_chess::BitBoard::DARK_SQUARES)
    }
    #[classattr]
    fn LIGHT_SQUARES() -> Self {
        BitBoard(cozy_chess::BitBoard::LIGHT_SQUARES)
    }

    // ── Constructors ───────────────────────────────────────────────────

    #[new]
    fn py_new(value: u64) -> Self {
        BitBoard(cozy_chess::BitBoard(value))
    }

    #[staticmethod]
    fn from_square(sq: &Square) -> Self {
        BitBoard(cozy_chess::BitBoard::from(sq.0))
    }

    #[staticmethod]
    fn from_file(f: &crate::enums::File) -> Self {
        BitBoard(cozy_chess::BitBoard::from(f.0))
    }

    #[staticmethod]
    fn from_rank(r: &crate::enums::Rank) -> Self {
        BitBoard(cozy_chess::BitBoard::from(r.0))
    }

    #[staticmethod]
    fn from_squares(squares: Vec<Square>) -> Self {
        let bb: cozy_chess::BitBoard = squares.iter().map(|s| s.0).collect();
        BitBoard(bb)
    }

    // ── Query methods ──────────────────────────────────────────────────

    fn has(&self, sq: &Square) -> bool {
        self.0.has(sq.0)
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn is_disjoint(&self, other: &Self) -> bool {
        self.0.is_disjoint(other.0)
    }

    fn is_subset(&self, other: &Self) -> bool {
        self.0.is_subset(other.0)
    }

    fn is_superset(&self, other: &Self) -> bool {
        self.0.is_superset(other.0)
    }

    fn next_square(&self) -> Option<Square> {
        self.0.next_square().map(Square)
    }

    fn flip_ranks(&self) -> Self {
        BitBoard(self.0.flip_ranks())
    }

    fn flip_files(&self) -> Self {
        BitBoard(self.0.flip_files())
    }

    fn squares(&self) -> Vec<Square> {
        self.0.into_iter().map(Square).collect()
    }

    // ── Bitwise operators ──────────────────────────────────────────────

    fn __and__(&self, other: &Self) -> Self {
        BitBoard(self.0 & other.0)
    }
    fn __or__(&self, other: &Self) -> Self {
        BitBoard(self.0 | other.0)
    }
    fn __xor__(&self, other: &Self) -> Self {
        BitBoard(self.0 ^ other.0)
    }
    fn __sub__(&self, other: &Self) -> Self {
        BitBoard(self.0 - other.0)
    }
    fn __invert__(&self) -> Self {
        BitBoard(!self.0)
    }

    // ── Sequence protocol ──────────────────────────────────────────────

    fn __len__(&self) -> usize {
        self.0.len() as usize
    }

    fn __contains__(&self, sq: &Square) -> bool {
        self.0.has(sq.0)
    }

    fn __iter__(&self) -> BitBoardIter {
        BitBoardIter {
            squares: self.0.into_iter().map(Square).collect(),
            index: 0,
        }
    }

    fn __bool__(&self) -> bool {
        !self.0.is_empty()
    }

    // ── Conversion ─────────────────────────────────────────────────────

    fn __int__(&self) -> u64 {
        self.0 .0
    }

    fn __repr__(&self) -> String {
        format!("BitBoard(0x{:016x})", self.0 .0)
    }

    fn __str__(&self) -> String {
        let mut s = String::new();
        for rank_idx in (0u8..8).rev() {
            for file_idx in 0u8..8 {
                let sq = cozy_chess::Square::new(
                    cozy_chess::File::index(file_idx as usize),
                    cozy_chess::Rank::index(rank_idx as usize),
                );
                if self.0.has(sq) {
                    s.push('X');
                } else {
                    s.push('.');
                }
                if file_idx < 7 {
                    s.push(' ');
                }
            }
            s.push('\n');
        }
        s
    }
}

// ── BitBoardIter ───────────────────────────────────────────────────────────

#[pyclass]
pub struct BitBoardIter {
    squares: Vec<Square>,
    index: usize,
}

#[pymethods]
impl BitBoardIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<Square> {
        if self.index < self.squares.len() {
            let sq = self.squares[self.index];
            self.index += 1;
            Some(sq)
        } else {
            None
        }
    }
}

// ── Register ───────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BitBoard>()?;
    m.add_class::<BitBoardIter>()?;
    Ok(())
}
