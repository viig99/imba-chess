use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ── Color ──────────────────────────────────────────────────────────────────

#[pyclass(frozen, eq, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Color(pub cozy_chess::Color);

#[pymethods]
impl Color {
    #[classattr]
    fn White() -> Self {
        Color(cozy_chess::Color::White)
    }
    #[classattr]
    fn Black() -> Self {
        Color(cozy_chess::Color::Black)
    }

    fn __invert__(&self) -> Self {
        Color(!self.0)
    }

    fn __repr__(&self) -> String {
        format!("Color.{:?}", self.0)
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
}

// ── Piece ──────────────────────────────────────────────────────────────────

#[pyclass(frozen, eq, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece(pub cozy_chess::Piece);

#[pymethods]
impl Piece {
    #[classattr]
    fn Pawn() -> Self {
        Piece(cozy_chess::Piece::Pawn)
    }
    #[classattr]
    fn Knight() -> Self {
        Piece(cozy_chess::Piece::Knight)
    }
    #[classattr]
    fn Bishop() -> Self {
        Piece(cozy_chess::Piece::Bishop)
    }
    #[classattr]
    fn Rook() -> Self {
        Piece(cozy_chess::Piece::Rook)
    }
    #[classattr]
    fn Queen() -> Self {
        Piece(cozy_chess::Piece::Queen)
    }
    #[classattr]
    fn King() -> Self {
        Piece(cozy_chess::Piece::King)
    }

    #[classattr]
    const NUM: usize = cozy_chess::Piece::NUM;

    #[classattr]
    fn ALL() -> Vec<Piece> {
        cozy_chess::Piece::ALL.iter().map(|&p| Piece(p)).collect()
    }

    fn __repr__(&self) -> String {
        format!("Piece.{:?}", self.0)
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    fn __index__(&self) -> usize {
        self.0 as usize
    }
}

// ── File ───────────────────────────────────────────────────────────────────

#[pyclass(frozen, eq, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct File(pub cozy_chess::File);

#[pymethods]
impl File {
    #[classattr]
    fn A() -> Self {
        File(cozy_chess::File::A)
    }
    #[classattr]
    fn B() -> Self {
        File(cozy_chess::File::B)
    }
    #[classattr]
    fn C() -> Self {
        File(cozy_chess::File::C)
    }
    #[classattr]
    fn D() -> Self {
        File(cozy_chess::File::D)
    }
    #[classattr]
    fn E() -> Self {
        File(cozy_chess::File::E)
    }
    #[classattr]
    fn F() -> Self {
        File(cozy_chess::File::F)
    }
    #[classattr]
    fn G() -> Self {
        File(cozy_chess::File::G)
    }
    #[classattr]
    fn H() -> Self {
        File(cozy_chess::File::H)
    }

    #[classattr]
    const NUM: usize = cozy_chess::File::NUM;

    #[classattr]
    fn ALL() -> Vec<File> {
        cozy_chess::File::ALL.iter().map(|&f| File(f)).collect()
    }

    fn __repr__(&self) -> String {
        format!("File.{:?}", self.0)
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
    fn __index__(&self) -> usize {
        self.0 as usize
    }
}

// ── Rank ───────────────────────────────────────────────────────────────────

#[pyclass(frozen, eq, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rank(pub cozy_chess::Rank);

#[pymethods]
impl Rank {
    #[classattr]
    fn First() -> Self {
        Rank(cozy_chess::Rank::First)
    }
    #[classattr]
    fn Second() -> Self {
        Rank(cozy_chess::Rank::Second)
    }
    #[classattr]
    fn Third() -> Self {
        Rank(cozy_chess::Rank::Third)
    }
    #[classattr]
    fn Fourth() -> Self {
        Rank(cozy_chess::Rank::Fourth)
    }
    #[classattr]
    fn Fifth() -> Self {
        Rank(cozy_chess::Rank::Fifth)
    }
    #[classattr]
    fn Sixth() -> Self {
        Rank(cozy_chess::Rank::Sixth)
    }
    #[classattr]
    fn Seventh() -> Self {
        Rank(cozy_chess::Rank::Seventh)
    }
    #[classattr]
    fn Eighth() -> Self {
        Rank(cozy_chess::Rank::Eighth)
    }

    #[classattr]
    const NUM: usize = cozy_chess::Rank::NUM;

    #[classattr]
    fn ALL() -> Vec<Rank> {
        cozy_chess::Rank::ALL.iter().map(|&r| Rank(r)).collect()
    }

    fn __repr__(&self) -> String {
        format!("Rank.{:?}", self.0)
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
    fn __index__(&self) -> usize {
        self.0 as usize
    }
}

// ── Square ─────────────────────────────────────────────────────────────────

#[pyclass(frozen, eq, hash, ord)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Square(pub cozy_chess::Square);

macro_rules! square_classattrs {
    ($($name:ident),*) => {
        #[pymethods]
        impl Square {
            $(
                #[classattr]
                fn $name() -> Self { Square(cozy_chess::Square::$name) }
            )*

            #[classattr] const NUM: usize = cozy_chess::Square::NUM;

            #[classattr]
            fn ALL() -> Vec<Square> {
                cozy_chess::Square::ALL.iter().map(|&s| Square(s)).collect()
            }

            #[staticmethod]
            fn new(file: &File, rank: &Rank) -> Self {
                Square(cozy_chess::Square::new(file.0, rank.0))
            }

            #[staticmethod]
            fn from_index(index: usize) -> PyResult<Self> {
                cozy_chess::Square::try_index(index)
                    .map(Square)
                    .ok_or_else(|| PyValueError::new_err("Square index out of range"))
            }

            #[staticmethod]
            fn from_str(s: &str) -> PyResult<Self> {
                s.parse::<cozy_chess::Square>()
                    .map(Square)
                    .map_err(|e| PyValueError::new_err(format!("{}", e)))
            }

            fn file(&self) -> File { File(self.0.file()) }
            fn rank(&self) -> Rank { Rank(self.0.rank()) }

            fn bitboard(&self) -> super::bitboard::BitBoard {
                super::bitboard::BitBoard(self.0.bitboard())
            }

            fn offset(&self, file_offset: i8, rank_offset: i8) -> PyResult<Self> {
                self.0.try_offset(file_offset, rank_offset)
                    .map(Square)
                    .ok_or_else(|| PyValueError::new_err("Offset out of bounds"))
            }

            fn try_offset(&self, file_offset: i8, rank_offset: i8) -> Option<Self> {
                self.0.try_offset(file_offset, rank_offset).map(Square)
            }

            fn flip_file(&self) -> Self { Square(self.0.flip_file()) }
            fn flip_rank(&self) -> Self { Square(self.0.flip_rank()) }

            fn relative_to(&self, color: &Color) -> Self {
                Square(self.0.relative_to(color.0))
            }

            fn __repr__(&self) -> String { format!("Square.{:?}", self.0) }
            fn __str__(&self) -> String { format!("{}", self.0) }
            fn __index__(&self) -> usize { self.0 as usize }
        }
    }
}

square_classattrs!(
    A1, B1, C1, D1, E1, F1, G1, H1, A2, B2, C2, D2, E2, F2, G2, H2, A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4, A5, B5, C5, D5, E5, F5, G5, H5, A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7, A8, B8, C8, D8, E8, F8, G8, H8
);

// ── GameStatus ─────────────────────────────────────────────────────────────

#[pyclass(frozen, eq, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GameStatus(pub cozy_chess::GameStatus);

#[pymethods]
impl GameStatus {
    #[classattr]
    fn Ongoing() -> Self {
        GameStatus(cozy_chess::GameStatus::Ongoing)
    }
    #[classattr]
    fn Won() -> Self {
        GameStatus(cozy_chess::GameStatus::Won)
    }
    #[classattr]
    fn Drawn() -> Self {
        GameStatus(cozy_chess::GameStatus::Drawn)
    }

    fn __repr__(&self) -> String {
        format!("GameStatus.{:?}", self.0)
    }
    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }
}

// ── Register ───────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Color>()?;
    m.add_class::<Piece>()?;
    m.add_class::<File>()?;
    m.add_class::<Rank>()?;
    m.add_class::<Square>()?;
    m.add_class::<GameStatus>()?;
    Ok(())
}
