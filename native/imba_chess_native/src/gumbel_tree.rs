//! Native Gumbel simulation loop: the tree walk of gumbel_search.gumbel_stepwise.
//!
//! Python used to own the tree (one `_Node` per evaluated position, a children
//! dict, the path list and the WDL sums) and crossed into Rust several times
//! per simulation for selection, edge expansion, node statistics and backup.
//! Here the whole tree lives natively: Python asks for the next leaf that needs
//! a network evaluation, scores it, and hands the evaluation back. Simulations
//! that end on a terminal or an already-expanded depth-limit node finish without
//! returning to Python.
//!
//! The arithmetic is the existing NodeStats code, called in the same order with
//! the same arguments, so search results are bit-identical to the Python loop.
//! Python keeps the network handles (opaque `extend()` results), root noise and
//! the final completed-Q, policy and WDL assembly.
//!
//! An optional `transition` callable replaces chess for synthetic environments
//! (the mctx goldens): positions are then opaque Python objects.

use pyo3::exceptions::{PyInterruptedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::board::Board;
use crate::chess_move::ChessMove;
use crate::gumbel::NodeStats;
use crate::leaf_eval::LeafEvaluation;
use crate::terminal::{repetition_hash, terminal, terminal_value};

const NONE: u32 = u32::MAX;

/// GumbelConfig, read once per search.
struct Config {
    simulations: usize,
    top_m: usize,
    max_depth: usize,
    maxvisit_init: f64,
    value_scale: f64,
    epsilon: f64,
    rescale_values: bool,
    root_forcing: bool,
    forcing_floor: bool,
    minimax_weight: f64,
    own_width: usize,
    own_forcing_floor: bool,
    visit_cap: u64,
}

impl Config {
    fn from_py(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            simulations: config.getattr("simulations")?.extract()?,
            top_m: config.getattr("top_m")?.extract()?,
            max_depth: config.getattr("max_depth")?.extract()?,
            maxvisit_init: config.getattr("maxvisit_init")?.extract()?,
            value_scale: config.getattr("value_scale")?.extract()?,
            epsilon: config.getattr("epsilon")?.extract()?,
            rescale_values: config.getattr("rescale_values")?.extract()?,
            root_forcing: config.getattr("root_forcing")?.extract()?,
            forcing_floor: config.getattr("forcing_floor")?.extract()?,
            minimax_weight: config.getattr("minimax_weight")?.extract()?,
            own_width: config.getattr("own_width")?.extract()?,
            own_forcing_floor: config.getattr("own_forcing_floor")?.extract()?,
            visit_cap: config.getattr("visit_cap")?.extract()?,
        })
    }

    fn floors(&self) -> bool {
        self.forcing_floor || self.own_forcing_floor
    }
}

/// mctx seq_halving.get_sequence_of_considered_visits (gumbel_search.considered_visits).
fn considered_visits(candidates: usize, simulations: usize) -> Vec<u64> {
    if candidates <= 1 {
        return (0..simulations as u64).collect();
    }
    // ceil(log2(candidates)), exact for integers.
    let rounds = (usize::BITS - (candidates - 1).leading_zeros()) as usize;
    let mut visits = vec![0u64; candidates];
    let mut sequence = Vec::with_capacity(simulations + candidates);
    let mut count = candidates;
    while sequence.len() < simulations {
        for _ in 0..(simulations / (rounds * count)).max(1) {
            sequence.extend_from_slice(&visits[..count]);
            for visit in &mut visits[..count] {
                *visit += 1;
            }
        }
        count = (count / 2).max(2);
    }
    sequence.truncate(simulations);
    sequence
}

/// The evaluation a node was initialized from.
enum Evaluation {
    /// Terminal position: no evaluation.
    Terminal,
    /// A decode-wave result; moves are read natively.
    Native(Py<LeafEvaluation>),
    /// Any other PositionEval-like object (root projections, test evaluators);
    /// moves are read through its move_at/uci_at/id_at, as the Python loop did.
    Python(PyObject),
}

/// A position: a chess board, or an opaque object of a synthetic environment.
enum State {
    Chess(cozy_chess::Board),
    Opaque(PyObject),
}

struct Node {
    board: State,
    // This position's repetition-hash history: a range of GumbelTree.histories.
    history: (usize, usize),
    handle: PyObject,
    value: f64,
    evaluation: Evaluation,
    wdl: Option<[f64; 3]>,
    stats: Option<NodeStats>,
    // Offset of this node's child indices (one per action) in
    // GumbelTree.child_slots; unused for terminal and pending nodes.
    children: usize,
}

/// The legal-projection error the Python _Node.initialize raised.
fn invalid_projection() -> PyErr {
    PyValueError::new_err("nonterminal evaluation has invalid legal projection")
}

/// _Node.initialize: statistics, value and WDL of one evaluated position.
fn initialize(
    evaluation: &Bound<'_, PyAny>,
    config: &Config,
    own: bool,
) -> PyResult<(Evaluation, NodeStats, f64, Option<[f64; 3]>)> {
    let floors = config.floors();
    let (kept, mut stats, value, wdl) = if let Ok(native) = evaluation.downcast::<LeafEvaluation>() {
        // Native decode results validate and build their stats in one call.
        let leaf = native.get();
        let stats = leaf.node_stats(config.minimax_weight, floors)?;
        (Evaluation::Native(native.clone().unbind()), stats, leaf.value, Some(leaf.wdl))
    } else {
        let n = evaluation.getattr("legal_ids")?.len()?;
        if n == 0
            || evaluation.getattr("legal_moves")?.len()? != n
            || evaluation.getattr("legal_ucis")?.len()? != n
            || evaluation.getattr("legal_log_priors")?.len()? != n
        {
            return Err(invalid_projection());
        }
        let priors: Vec<f64> = evaluation.getattr("legal_log_priors")?.extract()?;
        let value: f64 = evaluation.getattr("value_stm")?.extract()?;
        let wdl: Option<Vec<f64>> = evaluation.getattr("wdl")?.extract()?;
        // Native: finite value/prior and WDL validation, then the FLT_MIN-clamped
        // prior softmax, bit-identical to the former Python implementation.
        let mut stats =
            NodeStats::from_evaluation(value, priors, wdl.clone(), config.minimax_weight)?;
        let forcing = evaluation.getattr("legal_forcing")?;
        if forcing.len()? != n {
            return Err(invalid_projection());
        }
        if floors {
            stats.set_forcing(forcing.extract()?)?;
        }
        // from_evaluation accepted it, so a present WDL has exactly three entries.
        let wdl = wdl.map(|w| [w[0], w[1], w[2]]);
        (Evaluation::Python(evaluation.clone().unbind()), stats, value, wdl)
    };
    let n = stats.len();
    if config.own_width > 0 && own && n > config.own_width {
        let priors = stats.priors();
        let mut order: Vec<usize> = (0..n).collect();
        // sorted(range(n), key=lambda i: (-priors[i], i)); priors are finite.
        order.sort_by(|&a, &b| priors[b].partial_cmp(&priors[a]).unwrap().then(a.cmp(&b)));
        let mut keep = vec![false; n];
        for &i in &order[..config.own_width] {
            keep[i] = true;
        }
        stats.set_candidates(keep)?;
    }
    Ok((kept, stats, value, wdl))
}

/// A leaf waiting for its network evaluation.
struct Pending {
    child: u32,
    depth: usize,
}

#[pyclass(module = "imba_chess_native")]
pub struct GumbelTree {
    config: Config,
    transition: Option<PyObject>,
    nodes: Vec<Node>,
    // Flat per-tree storage, so a node owns no allocations of its own.
    histories: Vec<u64>,
    child_slots: Vec<u32>,
    schedule: Vec<u64>,
    next_visit: usize,
    path: Vec<(u32, usize)>,
    pending: Option<Pending>,
    evaluations: u64,
    terminal_hits: u64,
    cutoffs: u64,
    deepest: usize,
    wdl_sums: [f64; 3],
    edge_wdl_sums: Vec<[f64; 3]>,
    wdl_available: bool,
    considered: usize,
    // Measurement hook: called once per simulation (see set_observer).
    observer: Option<PyObject>,
}

impl GumbelTree {
    fn stats(&self, node: u32) -> &NodeStats {
        self.nodes[node as usize].stats.as_ref().expect("evaluated node")
    }

    fn root_action(&self, visit: u64) -> PyResult<usize> {
        let c = &self.config;
        self.stats(0)
            .root(visit, c.maxvisit_init, c.value_scale, c.epsilon, c.visit_cap, c.rescale_values)
    }

    /// The move of `action` at `node`.
    fn edge_move<'py>(&self, py: Python<'py>, node: u32, action: usize) -> PyResult<Bound<'py, PyAny>> {
        match &self.nodes[node as usize].evaluation {
            Evaluation::Native(leaf) => {
                let legal = leaf.get().legal();
                let mv = legal
                    .moves
                    .get(action)
                    .copied()
                    .ok_or_else(|| PyValueError::new_err("legal move index out of range"))?;
                Ok(Bound::new(py, ChessMove(mv))?.into_any())
            }
            Evaluation::Python(ev) => ev.bind(py).call_method1("move_at", (action,)),
            Evaluation::Terminal => unreachable!("terminal nodes are never expanded"),
        }
    }

    /// Play `action` at `node`: (child position, child history range in
    /// `histories`, terminal value).
    fn push(
        &mut self,
        py: Python<'_>,
        node: u32,
        action: usize,
    ) -> PyResult<(State, (usize, usize), Option<f64>)> {
        let parent = &self.nodes[node as usize];
        let (start, len) = parent.history;
        let child_start = self.histories.len();
        match (&self.transition, &parent.board) {
            (None, State::Chess(board)) => {
                let mv = match &parent.evaluation {
                    Evaluation::Native(leaf) => *leaf
                        .get()
                        .legal()
                        .moves
                        .get(action)
                        .ok_or_else(|| PyValueError::new_err("legal move index out of range"))?,
                    _ => self.edge_move(py, node, action)?.extract::<ChessMove>()?.0,
                };
                // push_and_classify, appending the child's history to the store.
                let mut child = board.clone();
                child.play_unchecked(mv);
                if child.halfmove_clock() != 0 {
                    self.histories.extend_from_within(start..start + len);
                    self.histories.push(repetition_hash(board));
                }
                let terminal = terminal_value(&child, true, &self.histories[child_start..]);
                Ok((State::Chess(child), (child_start, self.histories.len() - child_start), terminal))
            }
            (Some(transition), State::Opaque(board)) => {
                let mv = self.edge_move(py, node, action)?;
                let (child, history, terminal): (PyObject, Vec<u64>, Option<f64>) = transition
                    .bind(py)
                    .call1((board.clone_ref(py), mv, self.histories[start..start + len].to_vec(), true))?
                    .extract()?;
                self.histories.extend_from_slice(&history);
                Ok((State::Opaque(child), (child_start, history.len()), terminal))
            }
            _ => unreachable!("positions match the environment"),
        }
    }

    /// The UCI string and vocabulary id of `action` at `node`, for extend().
    fn edge_token(&self, py: Python<'_>, node: u32, action: usize) -> PyResult<(PyObject, PyObject)> {
        match &self.nodes[node as usize].evaluation {
            Evaluation::Native(leaf) => {
                let legal = leaf.get().legal();
                // edge_move already checked the index.
                let uci = PyString::new(py, &legal.ucis[action]).into_any().unbind();
                let id = legal.ids[action].into_pyobject(py)?.into_any().unbind();
                Ok((uci, id))
            }
            Evaluation::Python(ev) => {
                let ev = ev.bind(py);
                Ok((ev.call_method1("uci_at", (action,))?.unbind(), ev.call_method1("id_at", (action,))?.unbind()))
            }
            Evaluation::Terminal => unreachable!("terminal nodes are never expanded"),
        }
    }

    /// Back up `leaf` along the current path and add its WDL to the sums.
    fn complete(&mut self, py: Python<'_>, leaf: u32, depth: usize) -> PyResult<()> {
        let mut value = self.nodes[leaf as usize].value;
        if !value.is_finite() || value.abs() > 1.000001 {
            return Err(PyValueError::new_err("invalid backup value"));
        }
        for &(node, edge) in &self.path {
            if !self.stats(node).backup_ok(edge) {
                return Err(PyValueError::new_err("invalid backup edge or overflow"));
            }
        }
        // gumbel_backup: child_value starts from the leaf's own value.
        let mut child_value = value;
        for &(node, edge) in self.path.iter().rev() {
            value = -value;
            let stats = self.nodes[node as usize].stats.as_mut().expect("evaluated node");
            child_value = stats.backup_edge(edge, value, child_value);
        }
        let leaf = &self.nodes[leaf as usize];
        let leaf_wdl = match leaf.evaluation {
            Evaluation::Terminal => {
                let v = leaf.value;
                Some([f64::from(v == -1.0), f64::from(v == 0.0), f64::from(v == 1.0)])
            }
            _ => leaf.wdl,
        };
        if let Some(observer) = &self.observer {
            let terminal = matches!(leaf.evaluation, Evaluation::Terminal);
            let wdl = leaf_wdl.map(|w| (w[0], w[1], w[2]));
            observer.bind(py).call1((self.path[0].1, depth, wdl, terminal))?;
        }
        match leaf_wdl {
            None => self.wdl_available = false,
            Some(wdl) => {
                // Every chess ply swaps players; draws retain their perspective.
                let root_pov = if depth % 2 == 1 { [wdl[2], wdl[1], wdl[0]] } else { wdl };
                let edge = &mut self.edge_wdl_sums[self.path[0].1];
                for (i, probability) in root_pov.iter().enumerate() {
                    self.wdl_sums[i] += probability;
                    edge[i] += probability;
                }
            }
        }
        self.next_visit += 1;
        Ok(())
    }
}

#[pymethods]
impl GumbelTree {
    /// A tree over an evaluated, nonterminal root. `history` is the root's
    /// repetition-hash seed (search._root_hash_seed). transition, for synthetic
    /// environments only: (position, move, history, True) -> (child position,
    /// child history, terminal value or None), as push_and_classify.
    #[new]
    #[pyo3(signature = (config, board, history, handle, evaluation, transition=None))]
    fn new(
        config: &Bound<'_, PyAny>,
        board: &Bound<'_, PyAny>,
        history: Vec<u64>,
        handle: PyObject,
        evaluation: &Bound<'_, PyAny>,
        transition: Option<PyObject>,
    ) -> PyResult<Self> {
        let config = Config::from_py(config)?;
        let board = match transition {
            None => State::Chess(board.extract::<PyRef<'_, Board>>()?.0.clone()),
            Some(_) => State::Opaque(board.clone().unbind()),
        };
        let (evaluation, stats, value, wdl) = initialize(evaluation, &config, false)?;
        let n = stats.len();
        Ok(Self {
            config,
            transition,
            nodes: vec![Node {
                board,
                history: (0, history.len()),
                handle,
                value,
                evaluation,
                wdl,
                stats: Some(stats),
                children: 0,
            }],
            histories: history,
            child_slots: vec![NONE; n],
            schedule: Vec::new(),
            next_visit: 0,
            path: Vec::with_capacity(33),
            pending: None,
            evaluations: 1,
            terminal_hits: 0,
            cutoffs: 0,
            deepest: 0,
            wdl_sums: [0.0; 3],
            edge_wdl_sums: vec![[0.0; 3]; n],
            wdl_available: true,
            considered: 0,
            observer: None,
        })
    }

    /// Number of root actions.
    fn __len__(&self) -> usize {
        self.stats(0).len()
    }

    /// Root Gumbel noise, then the root candidates and the halving schedule.
    fn set_root_noise(&mut self, py: Python<'_>, noise: Vec<f64>) -> PyResult<()> {
        if !self.schedule.is_empty() {
            return Err(PyRuntimeError::new_err("root noise is already set"));
        }
        let n = self.stats(0).len();
        if noise.len() != n || noise.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err(
                "noise must contain one finite value per legal action",
            ));
        }
        let c = &self.config;
        let mut considered = c.top_m.min(n).min(c.simulations);
        let root = self.nodes[0].stats.as_mut().expect("evaluated root");
        root.set_noise(noise.clone())?;
        if c.root_forcing {
            // Root halving over the usual top_m (by noise + prior) plus every forcing move.
            let priors = root.priors();
            let keys: Vec<f64> = (0..n).map(|i| -(noise[i] + priors[i])).collect();
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| keys[a].partial_cmp(&keys[b]).unwrap().then(a.cmp(&b)));
            let mut candidates = vec![false; n];
            for &i in &order[..c.top_m.min(n)] {
                candidates[i] = true;
            }
            let forcing: Vec<bool> = match &self.nodes[0].evaluation {
                Evaluation::Native(leaf) => leaf.get().legal().forcing_flags().clone(),
                Evaluation::Python(ev) => ev.bind(py).getattr("legal_forcing")?.extract()?,
                Evaluation::Terminal => unreachable!("the root is evaluated"),
            };
            for (i, forcing) in forcing.iter().enumerate() {
                candidates[i] |= *forcing;
            }
            considered = candidates.iter().filter(|c| **c).count().min(c.simulations);
            self.nodes[0].stats.as_mut().expect("evaluated root").set_candidates(candidates)?;
        }
        self.considered = considered;
        self.schedule = considered_visits(considered, self.config.simulations);
        Ok(())
    }

    /// Root candidates of the sequential halving (after set_root_noise).
    #[getter]
    fn considered(&self) -> usize {
        self.considered
    }

    /// For measurement only: observer(root edge, depth, leaf WDL from the
    /// leaf's side to move or None, terminal) after every simulation's backup,
    /// before its WDL is added to the sums.
    fn set_observer(&mut self, observer: PyObject) {
        self.observer = Some(observer);
    }

    /// Run simulations until one reaches a new nonterminal position. Returns
    /// (parent handle, move UCI, move id, child board) for extend() and the
    /// network, or None when the budget is spent. should_stop() is checked
    /// before every simulation, as the Python loop did.
    fn next_leaf(
        &mut self,
        py: Python<'_>,
        should_stop: &Bound<'_, PyAny>,
    ) -> PyResult<Option<(PyObject, PyObject, PyObject, PyObject)>> {
        if self.pending.is_some() {
            return Err(PyRuntimeError::new_err("the previous leaf was not attached"));
        }
        if self.schedule.is_empty() && self.config.simulations > 0 {
            return Err(PyRuntimeError::new_err("root noise is not set"));
        }
        while self.next_visit < self.schedule.len() {
            if should_stop.call0()?.is_truthy()? {
                return Err(PyInterruptedError::new_err("search cancelled"));
            }
            let visit = self.schedule[self.next_visit];
            let (mut node, mut action, mut depth) = (0u32, self.root_action(visit)?, 0usize);
            self.path.clear();
            let leaf = loop {
                self.path.push((node, action));
                depth += 1;
                self.deepest = self.deepest.max(depth);
                let slot = self.nodes[node as usize].children + action;
                let child = self.child_slots[slot];
                if child == NONE {
                    let (board, history, terminal) = self.push(py, node, action)?;
                    let index = u32::try_from(self.nodes.len())
                        .ok()
                        .filter(|i| *i != NONE)
                        .ok_or_else(|| PyRuntimeError::new_err("search tree is full"))?;
                    self.child_slots[slot] = index;
                    match terminal {
                        Some(value) => {
                            self.nodes.push(Node {
                                board,
                                history,
                                handle: py.None(),
                                value,
                                evaluation: Evaluation::Terminal,
                                wdl: None,
                                stats: None,
                                children: usize::MAX,
                            });
                            self.terminal_hits += 1;
                            break index;
                        }
                        None => {
                            let (uci, id) = self.edge_token(py, node, action)?;
                            let parent_handle = self.nodes[node as usize].handle.clone_ref(py);
                            let child_board = match &board {
                                State::Chess(board) => Bound::new(py, Board(board.clone()))?.into_any().unbind(),
                                State::Opaque(board) => board.clone_ref(py),
                            };
                            self.nodes.push(Node {
                                board,
                                history,
                                handle: py.None(),
                                value: 0.0,
                                evaluation: Evaluation::Terminal,
                                wdl: None,
                                stats: None,
                                children: usize::MAX,
                            });
                            self.pending = Some(Pending { child: index, depth });
                            return Ok(Some((parent_handle, uci, id, child_board)));
                        }
                    }
                }
                node = child;
                let current = &self.nodes[node as usize];
                if let Evaluation::Terminal = current.evaluation {
                    self.terminal_hits += 1;
                    break node;
                }
                if depth == self.config.max_depth {
                    self.cutoffs += 1;
                    break node;
                }
                let c = &self.config;
                let floor = if depth % 2 == 1 { c.forcing_floor } else { c.own_forcing_floor };
                action = self.stats(node).interior(
                    c.maxvisit_init, c.value_scale, c.epsilon, floor, c.visit_cap, c.rescale_values,
                )?;
            };
            self.complete(py, leaf, depth)?;
        }
        Ok(None)
    }

    /// The network evaluation of the leaf next_leaf returned, with the handle
    /// extend() made for it; completes that simulation.
    fn attach(&mut self, py: Python<'_>, handle: PyObject, evaluation: &Bound<'_, PyAny>) -> PyResult<()> {
        let Pending { child, depth } = self
            .pending
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("no leaf is waiting for an evaluation"))?;
        // Opponent to move at odd depths; the root player's own nodes at even ones.
        let (kept, stats, value, wdl) = initialize(evaluation, &self.config, depth % 2 == 0)?;
        let n = stats.len();
        let node = &mut self.nodes[child as usize];
        node.handle = handle;
        node.evaluation = kept;
        node.stats = Some(stats);
        node.value = value;
        node.wdl = wdl;
        node.children = self.child_slots.len();
        self.child_slots.resize(self.child_slots.len() + n, NONE);
        self.evaluations += 1;
        if depth == self.config.max_depth {
            self.cutoffs += 1;
        }
        self.complete(py, child, depth)
    }

    /// After the last simulation: (visits, qvalues, prior probs, played action,
    /// evaluations, terminal hits, depth cutoffs, deepest, WDL sums, per-edge
    /// WDL sums, whether every leaf had a WDL, played child (handle, evaluation)
    /// or None). Terminal children have (None, None).
    #[allow(clippy::type_complexity)]
    fn finish(
        &self,
        py: Python<'_>,
    ) -> PyResult<(
        Vec<u64>,
        Vec<f64>,
        Vec<f64>,
        usize,
        u64,
        u64,
        u64,
        usize,
        (f64, f64, f64),
        Vec<(f64, f64, f64)>,
        bool,
        Option<(PyObject, PyObject)>,
    )> {
        if self.pending.is_some() || self.next_visit < self.schedule.len() {
            return Err(PyRuntimeError::new_err("search is not finished"));
        }
        let root = self.stats(0);
        let (visits, _, qvalues) = root.snapshot();
        let top = visits.iter().copied().max().unwrap_or(0);
        let action = self.root_action(top)?;
        let child = self.child_slots[self.nodes[0].children + action];
        let played = (child != NONE).then(|| {
            let node = &self.nodes[child as usize];
            let evaluation = match &node.evaluation {
                Evaluation::Terminal => py.None(),
                Evaluation::Native(leaf) => leaf.clone_ref(py).into_any(),
                Evaluation::Python(ev) => ev.clone_ref(py),
            };
            (node.handle.clone_ref(py), evaluation)
        });
        let s = self.wdl_sums;
        Ok((
            visits,
            qvalues,
            root.probs(),
            action,
            self.evaluations,
            self.terminal_hits,
            self.cutoffs,
            self.deepest,
            (s[0], s[1], s[2]),
            self.edge_wdl_sums.iter().map(|e| (e[0], e[1], e[2])).collect(),
            self.wdl_available,
            played,
        ))
    }

    /// Root visit counts so far (for tests and diagnostics).
    #[getter]
    fn root_visits(&self) -> Vec<u64> {
        self.stats(0).visits().to_vec()
    }

    /// Nodes created so far, terminal ones included.
    #[getter]
    fn size(&self) -> usize {
        self.nodes.len()
    }
}

/// Root-terminal check of the Python loop, on the root board and hash seed.
#[pyfunction]
fn gumbel_root_is_terminal(board: &Board, history: Vec<u64>) -> bool {
    terminal(&board.0, true, &history).is_some()
}

/// One node's initialization as the tree performs it: (stats, value). own
/// marks the root player's interior nodes (where own_width applies).
#[pyfunction]
fn _gumbel_node_stats(
    evaluation: &Bound<'_, PyAny>,
    config: &Bound<'_, PyAny>,
    own: bool,
) -> PyResult<(NodeStats, f64)> {
    let (_, stats, value, _) = initialize(evaluation, &Config::from_py(config)?, own)?;
    Ok((stats, value))
}

// Exposed for parity tests against gumbel_search.considered_visits.
#[pyfunction]
fn _considered_visits(candidates: usize, simulations: usize) -> Vec<u64> {
    considered_visits(candidates, simulations)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GumbelTree>()?;
    m.add_function(wrap_pyfunction!(gumbel_root_is_terminal, m)?)?;
    m.add_function(wrap_pyfunction!(_considered_visits, m)?)?;
    m.add_function(wrap_pyfunction!(_gumbel_node_stats, m)?)?;
    Ok(())
}
