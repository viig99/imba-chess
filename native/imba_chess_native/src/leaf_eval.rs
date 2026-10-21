//! Native leaf staging and results for the batched Gumbel decode wave.
//!
//! A decode wave used to turn every evaluated node into Python objects twice:
//! the legal projection returned four Python lists (ids, move wrappers, UCI
//! strings, forcing flags), and the readback became per-row float lists and a
//! PositionEval before NodeStats copied the priors back into Rust. Here the
//! projection stays native from staging through node initialization; Python
//! sees one object per node and reads single moves by index.
//!
//! Numerics are unchanged: the caller still computes values, WDL and log
//! priors with the same torch operations and passes the float32 results, which
//! are widened to f64 exactly as `.tolist()` did.

use std::sync::Arc;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::board::Board;
use crate::board_state::{encode, EncodeConfig};
use crate::chess_move::ChessMove;
use crate::gumbel::NodeStats;
use crate::move_projector::is_forcing;

/// One node's mapped legal moves in canonical UCI order (see MoveProjector).
#[pyclass(frozen)]
pub struct LegalSet {
    pub(crate) ids: Vec<i64>,
    pub(crate) moves: Vec<cozy_chess::Move>,
    pub(crate) ucis: Vec<Arc<str>>,
    pub(crate) forcing: std::sync::OnceLock<Vec<bool>>,
    pub(crate) board: cozy_chess::Board,
    pub(crate) total: usize,
}

impl LegalSet {
    /// Promotion, capture or check per move, computed once on first use.
    pub(crate) fn forcing_flags(&self) -> &Vec<bool> {
        self.forcing
            .get_or_init(|| self.moves.iter().map(|mv| is_forcing(&self.board, *mv)).collect())
    }
}

#[pymethods]
impl LegalSet {
    fn __len__(&self) -> usize {
        self.ids.len()
    }

    /// Legal moves before vocabulary mapping; unmapped moves are counted here.
    #[getter]
    fn total(&self) -> usize {
        self.total
    }

    #[getter]
    fn ids(&self) -> Vec<i64> {
        self.ids.clone()
    }

    #[getter]
    fn moves(&self) -> Vec<ChessMove> {
        self.moves.iter().map(|m| ChessMove(*m)).collect()
    }

    #[getter]
    fn ucis(&self) -> Vec<String> {
        self.ucis.iter().map(|u| u.to_string()).collect()
    }

    #[getter]
    fn forcing(&self) -> Vec<bool> {
        self.forcing_flags().clone()
    }
}

fn writable<'a>(
    py: Python<'a>,
    buffer: &'a PyBuffer<i64>,
) -> PyResult<&'a [std::cell::Cell<i64>]> {
    buffer
        .as_mut_slice(py)
        .ok_or_else(|| PyValueError::new_err("staging buffer must be writable and C-contiguous"))
}

fn readable<'a>(
    py: Python<'a>,
    buffer: &'a PyBuffer<f32>,
) -> PyResult<&'a [pyo3::buffer::ReadOnlyCell<f32>]> {
    buffer
        .as_slice(py)
        .ok_or_else(|| PyValueError::new_err("result buffer must be C-contiguous"))
}

/// Writes board features and legal ids of one wave into the packed host buffer.
#[pyclass(frozen)]
pub struct WaveStager {
    cfg: EncodeConfig,
    event_token: i64,
}

#[pymethods]
impl WaveStager {
    #[new]
    #[pyo3(signature = (
        ep_mode, halfmove_max, halfmove_bucket_size, fullmove_max,
        fullmove_bucket_size, event_token,
    ))]
    fn new(
        ep_mode: u8,
        halfmove_max: i64,
        halfmove_bucket_size: i64,
        fullmove_max: i64,
        fullmove_bucket_size: i64,
        event_token: i64,
    ) -> PyResult<Self> {
        Ok(Self {
            cfg: EncodeConfig::new(
                ep_mode,
                halfmove_max,
                halfmove_bucket_size,
                fullmove_max,
                fullmove_bucket_size,
            )?,
            event_token,
        })
    }

    /// Everything DecodeWorkspace.prepare writes per row, in one call. Per row:
    /// 64 piece ids at row * 64; the event token, five board scalars and the
    /// node's move id at scalars + k * pitch + row (k = 0..7); position (prefix
    /// length + depth), prefix length, ancestor count and destination row at
    /// positions + k * pitch + row (k = 0..4); the parent's arena chain at
    /// ancestors + row * ancestor_capacity; legal ids at legal + row *
    /// legal_capacity. Unwritten entries keep their value. Raises like the
    /// Python staging did when a parent was never decoded or the chain does not
    /// match the node's depth.
    #[allow(clippy::too_many_arguments)]
    fn stage_wave(
        &self,
        py: Python<'_>,
        host: &Bound<'_, PyAny>,
        nodes: Vec<Bound<'_, PyAny>>,
        boards: Vec<PyRef<'_, Board>>,
        legal: Vec<PyRef<'_, LegalSet>>,
        prefix_lens: Vec<i64>,
        dests: Vec<i64>,
        layout: WaveLayout,
    ) -> PyResult<()> {
        let g = boards.len();
        let l = &layout;
        if [nodes.len(), legal.len(), prefix_lens.len(), dests.len()].iter().any(|n| *n != g)
            || l.pitch < g
        {
            return Err(PyValueError::new_err("invalid wave staging layout"));
        }
        let buffer = PyBuffer::<i64>::get(host)?;
        let out = writable(py, &buffer)?;
        let needed = [
            64 * g,
            l.scalars + 6 * l.pitch + g,
            l.positions + 3 * l.pitch + g,
            l.ancestors + l.ancestor_capacity * g,
            l.legal + l.legal_capacity * g,
        ];
        if needed.iter().any(|n| *n > out.len()) || l.scalars < 64 * g {
            return Err(PyValueError::new_err("staging buffer is too small"));
        }
        let (parent_name, depth_name, move_name, chain_name) = (
            pyo3::intern!(py, "parent"),
            pyo3::intern!(py, "depth"),
            pyo3::intern!(py, "move_id"),
            pyo3::intern!(py, "arena_chain"),
        );
        for row in 0..g {
            let node = &nodes[row];
            let depth: usize = node.getattr(depth_name)?.extract()?;
            let parent = node.getattr(parent_name)?;
            let chain = if parent.is_none() {
                None
            } else {
                let chain = parent.getattr(chain_name)?;
                if chain.is_none() {
                    return Err(PyRuntimeError::new_err(
                        "Cannot decode a child before evaluating its parent",
                    ));
                }
                Some(chain)
            };
            let count = match &chain {
                None => 0,
                Some(chain) => chain.len()?,
            };
            if count != depth || count > l.ancestor_capacity {
                return Err(PyValueError::new_err(
                    "search suffix exceeds capacity or has invalid ancestry",
                ));
            }
            if let Some(chain) = chain {
                let base = l.ancestors + row * l.ancestor_capacity;
                if let Ok(list) = chain.downcast::<PyList>() {
                    for (j, item) in list.iter().enumerate() {
                        out[base + j].set(item.extract()?);
                    }
                } else {
                    for (j, item) in chain.try_iter()?.enumerate() {
                        out[base + j].set(item?.extract()?);
                    }
                }
            }
            let legal = &legal[row];
            if legal.ids.len() > l.legal_capacity {
                return Err(PyValueError::new_err("legal ids exceed staging capacity"));
            }
            let (pieces, scalars) = encode(&boards[row].0, &self.cfg);
            for (square, id) in pieces.iter().enumerate() {
                out[row * 64 + square].set(i64::from(*id));
            }
            out[l.scalars + row].set(self.event_token);
            for (k, value) in scalars.iter().enumerate() {
                out[l.scalars + (k + 1) * l.pitch + row].set(*value);
            }
            out[l.scalars + 6 * l.pitch + row].set(node.getattr(move_name)?.extract()?);
            let prefix = prefix_lens[row];
            out[l.positions + row].set(prefix + depth as i64);
            out[l.positions + l.pitch + row].set(prefix);
            out[l.positions + 2 * l.pitch + row].set(count as i64);
            out[l.positions + 3 * l.pitch + row].set(dests[row]);
            let base = l.legal + row * l.legal_capacity;
            for (j, id) in legal.ids.iter().enumerate() {
                out[base + j].set(*id);
            }
        }
        Ok(())
    }
}

/// Field offsets of the packed decode-wave buffer (DecodeWorkspace._fields).
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub struct WaveLayout {
    pitch: usize,
    scalars: usize,
    positions: usize,
    ancestors: usize,
    ancestor_capacity: usize,
    legal: usize,
    legal_capacity: usize,
}

/// After a decode wave: each node's arena chain becomes its parent's chain
/// plus the row its own K/V was written to.
#[pyfunction]
pub fn extend_arena_chains(
    py: Python<'_>,
    nodes: Vec<Bound<'_, PyAny>>,
    dests: Vec<i64>,
) -> PyResult<()> {
    if nodes.len() != dests.len() {
        return Err(PyValueError::new_err("one destination row per node"));
    }
    let (parent_name, chain_name) = (pyo3::intern!(py, "parent"), pyo3::intern!(py, "arena_chain"));
    for (node, dest) in nodes.iter().zip(dests) {
        let parent = node.getattr(parent_name)?;
        let chain = PyList::empty(py);
        if !parent.is_none() {
            for item in parent.getattr(chain_name)?.try_iter()? {
                chain.append(item?)?;
            }
        }
        chain.append(dest)?;
        node.setattr(chain_name, chain)?;
    }
    Ok(())
}

/// A network evaluation of one nonterminal leaf, ready to become search stats.
#[pyclass(frozen)]
pub struct LeafEvaluation {
    pub(crate) value: f64,
    pub(crate) wdl: [f64; 3],
    pub(crate) priors: Vec<f64>,
    // The node's projection, shared rather than copied.
    pub(crate) legal: Py<LegalSet>,
}

impl LeafEvaluation {
    pub(crate) fn legal(&self) -> &LegalSet {
        self.legal.get()
    }

    fn checked(&self, index: usize) -> PyResult<usize> {
        if index < self.legal().ids.len() {
            Ok(index)
        } else {
            Err(PyValueError::new_err("legal move index out of range"))
        }
    }
}

#[pymethods]
impl LeafEvaluation {
    fn __len__(&self) -> usize {
        self.legal().ids.len()
    }

    #[getter]
    fn value_stm(&self) -> f64 {
        self.value
    }

    #[getter]
    fn wdl(&self) -> (f64, f64, f64) {
        (self.wdl[0], self.wdl[1], self.wdl[2])
    }

    #[getter]
    fn legal_ids(&self) -> Vec<i64> {
        self.legal().ids.clone()
    }

    #[getter]
    fn legal_moves(&self) -> Vec<ChessMove> {
        self.legal().moves.iter().map(|m| ChessMove(*m)).collect()
    }

    #[getter]
    fn legal_ucis(&self) -> Vec<String> {
        self.legal().ucis.iter().map(|u| u.to_string()).collect()
    }

    #[getter]
    fn legal_log_priors(&self) -> Vec<f64> {
        self.priors.clone()
    }

    #[getter]
    fn legal_forcing(&self) -> Vec<bool> {
        self.legal().forcing_flags().clone()
    }

    fn move_at(&self, index: usize) -> PyResult<ChessMove> {
        Ok(ChessMove(self.legal().moves[self.checked(index)?]))
    }

    fn id_at(&self, index: usize) -> PyResult<i64> {
        Ok(self.legal().ids[self.checked(index)?])
    }

    fn uci_at(&self, index: usize) -> PyResult<String> {
        Ok(self.legal().ucis[self.checked(index)?].to_string())
    }

    /// NodeStats.from_evaluation over this evaluation, plus forcing flags
    /// when the search's forcing floors need them.
    #[pyo3(signature = (minimax_weight=0.0, with_forcing=false))]
    pub(crate) fn node_stats(&self, minimax_weight: f64, with_forcing: bool) -> PyResult<NodeStats> {
        let legal = self.legal();
        if legal.ids.is_empty() {
            return Err(PyValueError::new_err(
                "nonterminal evaluation has invalid legal projection",
            ));
        }
        NodeStats::from_parts(
            self.value,
            self.priors.clone(),
            Some(self.wdl.to_vec()),
            minimax_weight,
            with_forcing.then(|| legal.forcing_flags().clone()),
        )
    }
}

/// Leaf evaluations for one wave from float32 values [g], WDL [g, 3] and log
/// priors [g, width] (row-major), each row truncated to its legal count.
#[pyfunction]
pub fn make_evaluations(
    py: Python<'_>,
    legal: Vec<Bound<'_, LegalSet>>,
    values: &Bound<'_, PyAny>,
    wdls: &Bound<'_, PyAny>,
    priors: &Bound<'_, PyAny>,
    width: usize,
) -> PyResult<Vec<LeafEvaluation>> {
    let g = legal.len();
    let (values, wdls, priors) = (
        PyBuffer::<f32>::get(values)?,
        PyBuffer::<f32>::get(wdls)?,
        PyBuffer::<f32>::get(priors)?,
    );
    let (values, wdls, priors) = (
        readable(py, &values)?,
        readable(py, &wdls)?,
        readable(py, &priors)?,
    );
    if values.len() != g || wdls.len() != 3 * g || priors.len() != width * g {
        return Err(PyValueError::new_err("wave result shapes do not match"));
    }
    legal
        .iter()
        .enumerate()
        .map(|(row, bound)| {
            let legal = bound.get();
            let n = legal.ids.len();
            if n > width {
                return Err(PyValueError::new_err("legal ids exceed result width"));
            }
            Ok(LeafEvaluation {
                value: f64::from(values[row].get()),
                wdl: [
                    f64::from(wdls[3 * row].get()),
                    f64::from(wdls[3 * row + 1].get()),
                    f64::from(wdls[3 * row + 2].get()),
                ],
                priors: priors[row * width..row * width + n]
                    .iter()
                    .map(|p| f64::from(p.get()))
                    .collect(),
                legal: bound.clone().unbind(),
            })
        })
        .collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LegalSet>()?;
    m.add_class::<WaveStager>()?;
    m.add_class::<LeafEvaluation>()?;
    m.add_function(wrap_pyfunction!(make_evaluations, m)?)?;
    m.add_function(wrap_pyfunction!(extend_arena_chains, m)?)?;
    Ok(())
}
