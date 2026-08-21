use pyo3::prelude::*;

use crate::enums::File;

// ── CastleRights ───────────────────────────────────────────────────────────

#[pyclass(frozen, eq)]
#[derive(Clone, PartialEq)]
pub struct CastleRights(pub cozy_chess::CastleRights);

#[pymethods]
impl CastleRights {
    #[getter]
    fn short(&self) -> Option<File> {
        self.0.short.map(File)
    }

    #[getter]
    fn long(&self) -> Option<File> {
        self.0.long.map(File)
    }

    fn has_short(&self) -> bool {
        self.0.short.is_some()
    }
    fn has_long(&self) -> bool {
        self.0.long.is_some()
    }

    fn __repr__(&self) -> String {
        format!(
            "CastleRights(short={}, long={})",
            self.0
                .short
                .map_or("None".to_string(), |f| format!("File.{:?}", f)),
            self.0
                .long
                .map_or("None".to_string(), |f| format!("File.{:?}", f)),
        )
    }
}

// ── Register ───────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CastleRights>()?;
    Ok(())
}
