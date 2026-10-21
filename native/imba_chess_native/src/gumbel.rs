//! Native-owned search statistics. Python retains tree ownership and RNG.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// fsum's partial sums: on the stack, spilling to the heap only past 32
/// (non-overlapping partials of real search sums number a handful).
struct Partials {
    inline: [f64; 32],
    len: usize,
    heap: Option<Vec<f64>>,
}

impl Partials {
    fn new() -> Self {
        Self { inline: [0.0; 32], len: 0, heap: None }
    }

    fn slice(&mut self) -> &mut [f64] {
        match &mut self.heap {
            Some(heap) => heap,
            None => &mut self.inline[..self.len],
        }
    }

    fn truncate(&mut self, len: usize) {
        match &mut self.heap {
            Some(heap) => heap.truncate(len),
            None => self.len = len,
        }
    }

    fn push(&mut self, x: f64) {
        match &mut self.heap {
            Some(heap) => heap.push(x),
            None if self.len < self.inline.len() => {
                self.inline[self.len] = x;
                self.len += 1;
            }
            None => {
                let mut heap = self.inline.to_vec();
                heap.push(x);
                self.heap = Some(heap);
            }
        }
    }

    fn pop(&mut self) -> Option<f64> {
        match &mut self.heap {
            Some(heap) => heap.pop(),
            None if self.len > 0 => {
                self.len -= 1;
                Some(self.inline[self.len])
            }
            None => None,
        }
    }
}

/// math.fsum: the correctly rounded (half-even) sum. Search sums (prior
/// probabilities, softmax weights, probability-weighted Q) take the exact
/// fixed-point path; anything else falls back to expansion summation.
fn fsum<I: Iterator<Item = f64> + Clone>(values: I) -> f64 {
    match fixed_point_sum(values.clone()) {
        Some(sum) => sum,
        None => expansion_sum(values),
    }
}

/// Exact sum of terms with |x| <= 1 (at most 1024 of them) as an integer
/// multiple of 2^-116, rounded once: an i128 cast to f64 rounds half-even,
/// and scaling by a power of two is exact here. A term below 2^-64 is not a
/// multiple of 2^-116, so it only widens a bound of 2^52 units around the
/// exact total of the others; the result stands when both ends of that bound
/// round to the same double (rounding is monotone). None otherwise.
fn fixed_point_sum(values: impl Iterator<Item = f64>) -> Option<f64> {
    const SCALE: i32 = 116;
    let (mut total, mut slack, mut count) = (0i128, 0i128, 0usize);
    for x in values {
        count += 1;
        // Also rejects NaN.
        if count > 1024 || !(x.abs() <= 1.0) {
            return None;
        }
        if x == 0.0 {
            continue;
        }
        let bits = x.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        // x = mantissa * 2^(exponent - 1075) for normal x.
        let shift = exponent - 1075 + SCALE;
        if exponent == 0 || shift < 0 {
            slack += 1i128 << 52;
            continue;
        }
        let mantissa = ((bits & ((1u64 << 52) - 1)) | (1u64 << 52)) as i128;
        let units = mantissa << shift;
        total += if bits >> 63 == 1 { -units } else { units };
    }
    let unit = f64::from_bits(((1023 - SCALE) as u64) << 52);
    if slack == 0 {
        return Some(total as f64 * unit);
    }
    let (low, high) = ((total - slack) as f64, (total + slack) as f64);
    (low == high).then(|| low * unit)
}

// Expansion summation, including the final half-even correction. Ordinary
// compensated running sums do not reproduce math.fsum on cancellation/ties.
fn expansion_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials = Partials::new();
    for mut x in values {
        let mut i = 0;
        let current = partials.slice();
        for j in 0..current.len() {
            let mut y = current[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                current[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        if x != 0.0 {
            partials.push(x);
        }
    }
    let mut hi = partials.pop().unwrap_or(0.0);
    let mut lo = 0.0;
    while let Some(y) = partials.pop() {
        let x = hi;
        hi = x + y;
        lo = y - (hi - x);
        if lo != 0.0 {
            break;
        }
    }
    if let Some(&tail) = partials.slice().last() {
        if (lo < 0.0 && tail < 0.0) || (lo > 0.0 && tail > 0.0) {
            let y = lo * 2.0;
            let x = hi + y;
            if x - hi == y {
                hi = x;
            }
        }
    }
    hi
}

#[allow(clippy::too_many_arguments)]
fn completed(
    value: f64,
    visits: &[u64],
    qvalues: &[f64],
    probs: &[f64],
    maxvisit_init: f64,
    value_scale: f64,
    epsilon: f64,
    visit_cap: u64,
    rescale_values: bool,
) -> PyResult<Vec<f64>> {
    let mut values = Vec::with_capacity(visits.len());
    completed_into(
        value, visits, qvalues, probs, maxvisit_init, value_scale, epsilon, visit_cap,
        rescale_values, &mut values,
    )?;
    Ok(values)
}

/// completed() into a reused buffer.
#[allow(clippy::too_many_arguments)]
fn completed_into(
    value: f64,
    visits: &[u64],
    qvalues: &[f64],
    probs: &[f64],
    maxvisit_init: f64,
    value_scale: f64,
    epsilon: f64,
    visit_cap: u64,
    rescale_values: bool,
    values: &mut Vec<f64>,
) -> PyResult<()> {
    if visits.is_empty()
        || visits.len() != qvalues.len()
        || visits.len() != probs.len()
        || !value.is_finite()
        || qvalues.iter().any(|x| !x.is_finite())
        || probs.iter().any(|x| !x.is_finite() || *x < 0.0)
        || !maxvisit_init.is_finite()
        || maxvisit_init < 0.0
        || !value_scale.is_finite()
        || value_scale < 0.0
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err(PyValueError::new_err("invalid completed-Q inputs"));
    }
    let count = visits
        .iter()
        .try_fold(0u64, |a, b| a.checked_add(*b))
        .ok_or_else(|| PyValueError::new_err("visit count overflow"))?;
    let mass = fsum(
        probs
            .iter()
            .zip(visits)
            .filter_map(|(p, n)| (*n != 0).then_some(*p)),
    );
    let weighted = if mass != 0.0 {
        fsum(
            probs
                .iter()
                .zip(qvalues)
                .zip(visits)
                .filter_map(|((p, q), n)| (*n != 0).then_some(p * q / mass)),
        )
    } else {
        0.0
    };
    let mixed = (value + count as f64 * weighted) / ((count as u128 + 1) as f64);
    values.clear();
    values.extend(
        qvalues
            .iter()
            .zip(visits)
            .map(|(q, n)| if *n != 0 { *q } else { mixed }),
    );
    // visit_cap > 0 bounds the visit term so larger budgets deepen search without
    // growing the value bonus beyond its value at that visit count.
    let top = *visits.iter().max().unwrap();
    let top = if visit_cap > 0 {
        top.min(visit_cap)
    } else {
        top
    };
    let scale = (maxvisit_init + top as f64) * value_scale;
    // mctx rescale_values: min-max normalize per node; otherwise use raw Q.
    if rescale_values {
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator = (high - low).max(epsilon);
        for q in values.iter_mut() {
            *q = (*q - low) / denominator;
        }
    }
    for q in values.iter_mut() {
        *q *= scale;
    }
    Ok(())
}

/// Per-thread buffers for selection, so a simulation allocates nothing.
#[derive(Default)]
struct Scratch {
    values: Vec<f64>,
    weights: Vec<f64>,
    qvalues: Vec<f64>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

// CPython 3.12+ builtin sum() over floats (Neumaier compensation), for the
// WDL normalization check to agree with the Python reference at its boundary.
fn python_sum(values: &[f64]) -> f64 {
    let (mut total, mut c) = (0.0f64, 0.0f64);
    for &x in values {
        let t = total + x;
        if total.abs() >= x.abs() {
            c += (total - t) + x;
        } else {
            c += (x - t) + total;
        }
        total = t;
    }
    if c != 0.0 && c.is_finite() {
        total += c;
    }
    total
}

// gumbel_search.softmax clamped at FLT_MIN: max, exp(x - max), math.fsum, divide.
fn clamped_softmax_into(logits: &[f64], probs: &mut [f64]) {
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for (p, x) in probs.iter_mut().zip(logits) {
        *p = (x - maximum).exp();
    }
    let total = fsum(probs.iter().copied());
    for p in probs.iter_mut() {
        let q = *p / total;
        *p = if f32::MIN_POSITIVE as f64 > q { f32::MIN_POSITIVE as f64 } else { q };
    }
}

fn validate_priors(priors: &[f64], n: usize) -> PyResult<()> {
    if priors.len() != n || priors.iter().any(|p| !p.is_finite()) {
        return Err(PyValueError::new_err("invalid prior logits"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gumbel_interior_action(
    value: f64,
    priors: &[f64],
    visits: &[u64],
    qvalues: &[f64],
    prior_probs: &[f64],
    maxvisit_init: f64,
    value_scale: f64,
    epsilon: f64,
    candidates: &[bool],
    visit_cap: u64,
    rescale_values: bool,
    scratch: &mut Scratch,
) -> PyResult<usize> {
    validate_priors(&priors, visits.len())?;
    let Scratch { values: logits, weights, .. } = scratch;
    completed_into(
        value,
        &visits,
        &qvalues,
        &prior_probs,
        maxvisit_init,
        value_scale,
        epsilon,
        visit_cap,
        rescale_values,
        logits,
    )?;
    // Completed Q becomes the logits in place: prior + q.
    for (q, p) in logits.iter_mut().zip(priors) {
        *q = p + *q;
    }
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    weights.clear();
    weights.extend(logits.iter().map(|x| (x - maximum).exp()));
    let total = fsum(weights.iter().copied());
    let denominator = (visits.iter().map(|n| *n as u128).sum::<u128>() + 1) as f64;
    let mut action = 0;
    let mut best = f64::NEG_INFINITY;
    for (i, (weight, n)) in weights.iter().zip(visits).enumerate() {
        if !candidates.is_empty() && !candidates[i] {
            continue;
        }
        let score = weight / total - *n as f64 / denominator;
        if score > best {
            // Strict comparison preserves first-index ties.
            best = score;
            action = i;
        }
    }
    Ok(action)
}

#[allow(clippy::too_many_arguments)]
fn gumbel_root_action(
    value: f64,
    priors: &[f64],
    visits: &[u64],
    qvalues: &[f64],
    prior_probs: &[f64],
    maxvisit_init: f64,
    value_scale: f64,
    epsilon: f64,
    noise: &[f64],
    eligible_visit: u64,
    max_prior: f64,
    candidates: &[bool],
    visit_cap: u64,
    rescale_values: bool,
    scratch: &mut Scratch,
) -> PyResult<usize> {
    validate_priors(&priors, visits.len())?;
    if noise.len() != visits.len() || noise.iter().any(|x| !x.is_finite()) || !max_prior.is_finite()
    {
        return Err(PyValueError::new_err("invalid root noise"));
    }
    let q = &mut scratch.values;
    completed_into(
        value,
        &visits,
        &qvalues,
        &prior_probs,
        maxvisit_init,
        value_scale,
        epsilon,
        visit_cap,
        rescale_values,
        q,
    )?;
    let mut action = None;
    let mut best = f64::NEG_INFINITY;
    for i in 0..visits.len() {
        if visits[i] == eligible_visit && (candidates.is_empty() || candidates[i]) {
            let score = (-1e9f64).max(noise[i] + priors[i] - max_prior + q[i]);
            if score > best {
                best = score;
                action = Some(i);
            }
        }
    }
    action.ok_or_else(|| PyValueError::new_err("no eligible root action"))
}

// Columns of NodeStats.edges after the visit counts.
const PRIORS: usize = 1;
const PROBS: usize = 2;
const SUMS: usize = 3;
const MEANS: usize = 4;
const MINIMAX: usize = 5;
const COLUMNS: usize = 6;

fn as_f64(bits: &[u64]) -> &[f64] {
    // SAFETY: u64 and f64 have the same size and alignment, and every bit
    // pattern is a valid f64.
    unsafe { std::slice::from_raw_parts(bits.as_ptr().cast::<f64>(), bits.len()) }
}

fn as_f64_mut(bits: &mut [u64]) -> &mut [f64] {
    // SAFETY: as in as_f64; the borrow is unique.
    unsafe { std::slice::from_raw_parts_mut(bits.as_mut_ptr().cast::<f64>(), bits.len()) }
}

#[pyclass]
pub(crate) struct NodeStats {
    value: f64,
    // Per-edge arrays in one allocation (a tree drops hundreds of nodes per
    // search): visits, then the priors, probs, sums, means and edge negamax
    // columns as f64 bits. See col() and backup_edge().
    n: usize,
    edges: Box<[u64]>,
    noise: Vec<f64>,
    max_prior: f64,
    // Halving-style options (all inert by default; see GumbelConfig).
    minimax_weight: f64,
    forcing: Vec<bool>,
    candidates: Vec<bool>,
    // PUCT experiments are opt-in: the existing Gumbel arithmetic is unchanged.
    puct: Option<PuctParameters>,
    selection_probs: Vec<f64>,
    m2: Vec<f64>,
    variance_count: u64,
    variance_mean: f64,
    variance_m2: f64,
}

struct PuctParameters {
    mode: u8,
    cpuct: f64,
    base: f64,
    factor: f64,
    fpu: f64,
    c1: f64,
    c2: f64,
    prior_variance: f64,
    prior_weight: f64,
    // Opt-in paper/Lc0 forms; false keeps the original study arithmetic.
    lc0_parent_q: bool,
    paper_log_n: bool,
}

impl NodeStats {
    /// from_evaluation for native callers, optionally with forcing flags set.
    pub(crate) fn from_parts(
        value: f64,
        priors: &[f64],
        wdl: Option<&[f64]>,
        minimax_weight: f64,
        forcing: Option<Vec<bool>>,
    ) -> PyResult<Self> {
        let mut stats = Self::evaluated(value, priors, wdl, minimax_weight)?;
        if let Some(forcing) = forcing {
            stats.set_forcing(forcing)?;
        }
        Ok(stats)
    }

    /// from_evaluation over borrowed inputs: the clamped prior softmax is
    /// written straight into the node's arrays.
    fn evaluated(
        value: f64,
        priors: &[f64],
        wdl: Option<&[f64]>,
        minimax_weight: f64,
    ) -> PyResult<Self> {
        if !value.is_finite() || value.abs() > 1.000001 || priors.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err(
                "nonfinite or invalid network evaluation",
            ));
        }
        if let Some(wdl) = &wdl {
            if wdl.len() != 3
                || wdl.iter().any(|p| !p.is_finite() || *p < 0.0)
                || (python_sum(wdl) - 1.0).abs() > 1e-5
                || (wdl[2] - wdl[0] - value).abs() > 1e-5
            {
                return Err(PyValueError::new_err(
                    "invalid or inconsistent evaluation WDL",
                ));
            }
        }
        // The checks of new(); the softmax output is always valid probabilities.
        if priors.is_empty() {
            return Err(PyValueError::new_err("invalid node statistics"));
        }
        if !minimax_weight.is_finite() || !(0.0..=1.0).contains(&minimax_weight) {
            return Err(PyValueError::new_err("minimax_weight must be in [0, 1]"));
        }
        Ok(Self::allocate(value, priors.len(), minimax_weight, |p, q| {
            p.copy_from_slice(priors);
            clamped_softmax_into(priors, q);
        }))
    }

    /// Edge Q used for selection and reported to Python: the running mean, or
    /// (1 - w) * mean + w * negamax for visited edges when minimax_weight > 0.
    fn qvalues(&self) -> Vec<f64> {
        let mut out = Vec::new();
        self.qvalues_into(&mut out).to_vec()
    }

    /// qvalues() without a copy: the means themselves, or the blend in `out`.
    fn qvalues_into<'a>(&'a self, out: &'a mut Vec<f64>) -> &'a [f64] {
        let w = self.minimax_weight;
        if w == 0.0 {
            return &self.col(MEANS);
        }
        out.clear();
        out.extend(
            self.col(MEANS)
                .iter()
                .zip(self.col(MINIMAX))
                .zip(self.visits())
                .map(|((m, mm), n)| if *n != 0 { (1.0 - w) * m + w * mm } else { *m }),
        );
        out
    }

    /// Whether `edge` can take one more backup (validated before any mutation).
    pub(crate) fn backup_ok(&self, edge: usize) -> bool {
        edge < self.visits().len() && self.visits()[edge] != u64::MAX
    }

    /// One edge of a backup: `value` is already from this node's side to move,
    /// `child_value` is the side-to-move value of the node below the edge.
    /// Returns this node's side-to-move value for the edge above it.
    pub(crate) fn backup_edge(&mut self, edge: usize, value: f64, child_value: f64) -> f64 {
        let n = self.n;
        let (visits, columns) = self.edges.split_at_mut(n);
        let columns = as_f64_mut(columns);
        let (sums, rest) = columns[(SUMS - 1) * n..].split_at_mut(n);
        let (means, minimax) = rest.split_at_mut(n);
        let old_mean = means[edge];
        visits[edge] += 1;
        sums[edge] += value;
        means[edge] = sums[edge] / visits[edge] as f64;
        if !self.m2.is_empty() {
            self.m2[edge] += (value - old_mean) * (value - means[edge]);
            self.variance_count += 1;
            let delta = value - self.variance_mean;
            self.variance_mean += delta / self.variance_count as f64;
            self.variance_m2 += delta * (value - self.variance_mean);
        }
        minimax[edge] = -child_value;
        self.side_to_move_value()
    }

    pub(crate) fn len(&self) -> usize {
        self.n
    }

    fn col(&self, column: usize) -> &[f64] {
        as_f64(&self.edges[column * self.n..(column + 1) * self.n])
    }

    pub(crate) fn priors(&self) -> &[f64] {
        self.col(PRIORS)
    }

    pub(crate) fn visits(&self) -> &[u64] {
        &self.edges[..self.n]
    }

    /// Validated statistics with zeroed visits, sums, means and negamax
    /// values; priors and probs are written by `fill(priors, probs)`.
    fn allocate(
        value: f64,
        n: usize,
        minimax_weight: f64,
        fill: impl FnOnce(&mut [f64], &mut [f64]),
    ) -> Self {
        let mut edges = vec![0u64; COLUMNS * n].into_boxed_slice();
        let columns = as_f64_mut(&mut edges[n..]);
        let (priors, rest) = columns.split_at_mut(n);
        fill(priors, &mut rest[..n]);
        let max_prior = priors.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self {
            value,
            n,
            edges,
            noise: Vec::new(),
            max_prior,
            minimax_weight,
            forcing: Vec::new(),
            candidates: Vec::new(),
            puct: None,
            selection_probs: Vec::new(),
            m2: Vec::new(),
            variance_count: 0,
            variance_mean: 0.0,
            variance_m2: 0.0,
        }
    }

    /// Pure negamax value of this node, independent of the selection blend.
    /// Blending here as well would attenuate tactical evidence at every ply.
    fn side_to_move_value(&self) -> f64 {
        let mut best = f64::NEG_INFINITY;
        for (value, n) in self.col(MINIMAX).iter().zip(self.visits()) {
            if *n != 0 && *value > best {
                best = *value;
            }
        }
        if best == f64::NEG_INFINITY {
            self.value
        } else {
            best
        }
    }
}

#[pymethods]
impl NodeStats {
    #[new]
    #[pyo3(signature = (value, priors, probs, minimax_weight=0.0))]
    fn new(value: f64, priors: Vec<f64>, probs: Vec<f64>, minimax_weight: f64) -> PyResult<Self> {
        let n = priors.len();
        if n == 0
            || probs.len() != n
            || !value.is_finite()
            || priors.iter().any(|x| !x.is_finite())
            || probs.iter().any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(PyValueError::new_err("invalid node statistics"));
        }
        if !minimax_weight.is_finite() || !(0.0..=1.0).contains(&minimax_weight) {
            return Err(PyValueError::new_err("minimax_weight must be in [0, 1]"));
        }
        Ok(Self::allocate(value, n, minimax_weight, |p, q| {
            p.copy_from_slice(&priors);
            q.copy_from_slice(&probs);
        }))
    }

    /// A search node from one network evaluation: the validation and clamped
    /// prior softmax of the Python _Node.initialize, bit-for-bit, in one call.
    #[staticmethod]
    #[pyo3(signature = (value, priors, wdl=None, minimax_weight=0.0))]
    pub(crate) fn from_evaluation(
        value: f64,
        priors: Vec<f64>,
        wdl: Option<Vec<f64>>,
        minimax_weight: f64,
    ) -> PyResult<Self> {
        Self::evaluated(value, &priors, wdl.as_deref(), minimax_weight)
    }

    /// Clamped prior probabilities, as completed-Q expects them.
    #[getter]
    pub(crate) fn probs(&self) -> Vec<f64> {
        self.col(PROBS).to_vec()
    }

    /// mode 0: Lc0-style PUCT; 1: parent variance scaling; 2: PUCT-V.
    /// lc0_parent_q: FPU parent Q counts the node's own evaluation as a visit.
    /// paper_log_n: PUCT-V uses log N (Weichart Eq. 17 in v3) instead of ln(N + 1).
    #[pyo3(signature = (mode, cpuct, base, factor, fpu, c1, c2, prior_variance, prior_weight,
                        selection_probs=None, lc0_parent_q=false, paper_log_n=false))]
    #[allow(clippy::too_many_arguments)]
    fn configure_puct(
        &mut self, mode: u8, cpuct: f64, base: f64, factor: f64,
        fpu: f64, c1: f64, c2: f64, prior_variance: f64, prior_weight: f64,
        selection_probs: Option<Vec<f64>>, lc0_parent_q: bool, paper_log_n: bool,
    ) -> PyResult<()> {
        if mode > 2 || [cpuct, base, factor, fpu, c1, c2, prior_variance, prior_weight]
            .iter().any(|x| !x.is_finite() || *x < 0.0)
            || base == 0.0 || prior_variance == 0.0 || prior_weight == 0.0
            || self.visits().iter().any(|n| *n != 0)
        {
            return Err(PyValueError::new_err("invalid PUCT parameters or nonempty statistics"));
        }
        if let Some(probs) = selection_probs {
            if probs.len() != self.col(PROBS).len() || probs.iter().any(|p| !p.is_finite() || *p < 0.0)
                || (fsum(probs.iter().copied()) - 1.0).abs() > 1e-6
            {
                return Err(PyValueError::new_err("invalid PUCT selection priors"));
            }
            self.selection_probs = probs;
        }
        if mode != 0 {
            self.m2 = vec![0.0; self.col(PROBS).len()];
        }
        self.puct = Some(PuctParameters {
            mode, cpuct, base, factor, fpu, c1, c2, prior_variance, prior_weight, lc0_parent_q, paper_log_n,
        });
        Ok(())
    }

    fn select_puct(&self) -> PyResult<usize> {
        let cfg = self.puct.as_ref().ok_or_else(|| PyValueError::new_err("PUCT not configured"))?;
        let probs: &[f64] = if self.selection_probs.is_empty() { self.col(PROBS) } else { &self.selection_probs };
        let count = self.visits().iter().sum::<u64>() as f64;
        let parent_q = if cfg.lc0_parent_q {
            // Lc0's node Q: its own network value is the node's first visit.
            (self.value + self.col(SUMS).iter().sum::<f64>()) / (count + 1.0)
        } else if count > 0.0 {
            self.col(SUMS).iter().sum::<f64>() / count
        } else {
            self.value
        };
        // An empty node never reaches the bonus (see below), so log N stays finite.
        let log_n = if cfg.paper_log_n { count.max(1.0).ln() } else { (count + 1.0).ln() };
        let visited_mass: f64 = probs.iter().zip(self.visits()).filter_map(|(p, n)| (*n > 0).then_some(*p)).sum();
        let unvisited_q = parent_q - cfg.fpu * visited_mass.sqrt();
        let mut c = cfg.cpuct + cfg.factor * ((count + cfg.base + 1.0) / cfg.base).ln();
        if cfg.mode == 1 {
            let variance = (self.variance_m2 + cfg.prior_weight * cfg.prior_variance)
                / (self.variance_count as f64 + cfg.prior_weight);
            c *= (variance.max(0.0) / cfg.prior_variance).sqrt();
        }
        let mut best = f64::NEG_INFINITY;
        let mut selected = 0;
        for i in 0..probs.len() {
            let n = self.visits()[i] as f64;
            let q = if n > 0.0 { self.col(MEANS)[i] } else { unvisited_q };
            let bonus = if cfg.mode == 2 {
                // Weichart, arXiv:2512.21648 v1 Eq. 22 (v3 Eq. 17), with
                // ln(N + 1) unless paper_log_n. Rewards here span [-1,1],
                // so the default bias coefficient is twice the [0,1] one.
                // With fewer than two samples the empirical variance is zero;
                // the separate bias bonus still drives exploration.
                let stdev = if n > 0.0 { (self.m2[i] / n).max(0.0).sqrt() } else { 0.0 };
                probs[i] * (cfg.c1 * stdev * count.max(1.0).sqrt()
                    + cfg.c2 * log_n) / (1.0 + n)
            } else {
                c * probs[i] * count.max(1.0).sqrt() / (1.0 + n)
            };
            // At an empty node explicitly prefer the prior, including PUCT-V
            // whose empirical variance and log term are both zero.
            let score = if count == 0.0 { probs[i] } else { q + bonus };
            if score > best {
                best = score;
                selected = i;
            }
        }
        Ok(selected)
    }

    fn variance_snapshot(&self) -> (Vec<f64>, u64, f64, f64) {
        (self.m2.clone(), self.variance_count, self.variance_mean, self.variance_m2)
    }

    /// Forcing flags (capture, check, promotion) per legal move, for the floor.
    pub(crate) fn set_forcing(&mut self, forcing: Vec<bool>) -> PyResult<()> {
        if forcing.len() != self.col(PRIORS).len() {
            return Err(PyValueError::new_err("invalid forcing mask"));
        }
        self.forcing = forcing;
        Ok(())
    }

    /// Selectable actions: root halving candidates, or an interior width cap. Empty = all.
    pub(crate) fn set_candidates(&mut self, candidates: Vec<bool>) -> PyResult<()> {
        if candidates.len() != self.col(PRIORS).len() || !candidates.iter().any(|c| *c) {
            return Err(PyValueError::new_err("invalid candidate mask"));
        }
        self.candidates = candidates;
        Ok(())
    }

    pub(crate) fn set_noise(&mut self, noise: Vec<f64>) -> PyResult<()> {
        if noise.len() != self.col(PRIORS).len() || noise.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err("invalid root noise"));
        }
        self.noise = noise;
        Ok(())
    }

    #[pyo3(signature = (maxvisit_init, value_scale, epsilon, forcing_floor=false, visit_cap=0, rescale_values=false))]
    pub(crate) fn interior(
        &self,
        maxvisit_init: f64,
        value_scale: f64,
        epsilon: f64,
        forcing_floor: bool,
        visit_cap: u64,
        rescale_values: bool,
    ) -> PyResult<usize> {
        if forcing_floor {
            if self.forcing.len() != self.col(PRIORS).len() {
                return Err(PyValueError::new_err("forcing floor requires set_forcing"));
            }
            // Unvisited forcing replies first: highest prior, first index on ties.
            let mut pick: Option<usize> = None;
            for i in 0..self.col(PRIORS).len() {
                if self.forcing[i]
                    && self.visits()[i] == 0
                    && pick.map_or(true, |j| self.col(PRIORS)[i] > self.col(PRIORS)[j])
                {
                    pick = Some(i);
                }
            }
            if let Some(i) = pick {
                return Ok(i);
            }
        }
        SCRATCH.with_borrow_mut(|scratch| {
            let mut blended = std::mem::take(&mut scratch.qvalues);
            let result = gumbel_interior_action(
                self.value,
                &self.col(PRIORS),
                &self.visits(),
                self.qvalues_into(&mut blended),
                &self.col(PROBS),
                maxvisit_init,
                value_scale,
                epsilon,
                &self.candidates,
                visit_cap,
                rescale_values,
                scratch,
            );
            scratch.qvalues = blended;
            result
        })
    }

    #[pyo3(signature = (visit, maxvisit_init, value_scale, epsilon, visit_cap=0, rescale_values=false))]
    pub(crate) fn root(
        &self,
        visit: u64,
        maxvisit_init: f64,
        value_scale: f64,
        epsilon: f64,
        visit_cap: u64,
        rescale_values: bool,
    ) -> PyResult<usize> {
        SCRATCH.with_borrow_mut(|scratch| {
            let mut blended = std::mem::take(&mut scratch.qvalues);
            let result = gumbel_root_action(
                self.value,
                &self.col(PRIORS),
                &self.visits(),
                self.qvalues_into(&mut blended),
                &self.col(PROBS),
                maxvisit_init,
                value_scale,
                epsilon,
                &self.noise,
                visit,
                self.max_prior,
                &self.candidates,
                visit_cap,
                rescale_values,
                scratch,
            );
            scratch.qvalues = blended;
            result
        })
    }

    pub(crate) fn snapshot(&self) -> (Vec<u64>, Vec<f64>, Vec<f64>) {
        (self.visits().to_vec(), self.col(SUMS).to_vec(), self.qvalues())
    }
}

#[pyfunction]
fn gumbel_backup(
    py: Python<'_>,
    path: Vec<(Py<NodeStats>, usize)>,
    mut value: f64,
) -> PyResult<()> {
    if !value.is_finite() || value.abs() > 1.000001 {
        return Err(PyValueError::new_err("invalid backup value"));
    }
    // Validate the complete path before mutating any node. Real search paths
    // contain one edge per distinct node; reject aliases to preserve that rule.
    let mut seen = std::collections::HashSet::new();
    for (node, edge) in &path {
        if !seen.insert(node.as_ptr()) {
            return Err(PyValueError::new_err("duplicate backup node"));
        }
        let node = node.try_borrow(py)?;
        if !node.backup_ok(*edge) {
            return Err(PyValueError::new_err("invalid backup edge or overflow"));
        }
    }
    // child_value: side-to-move value of the node below the current edge,
    // starting from the leaf's own value; feeds the edge negamax values.
    let mut child_value = value;
    for (node, edge) in path.iter().rev() {
        value = -value;
        child_value = node.try_borrow_mut(py)?.backup_edge(*edge, value, child_value);
    }
    Ok(())
}

// Exposed for direct numerical parity tests; selectors never round-trip Q arrays.
#[pyfunction]
#[pyo3(signature = (value, visits, qvalues, prior_probs, maxvisit_init, value_scale, epsilon, visit_cap=0, rescale_values=false))]
#[allow(clippy::too_many_arguments)]
fn _gumbel_completed_q(
    value: f64,
    visits: Vec<u64>,
    qvalues: Vec<f64>,
    prior_probs: Vec<f64>,
    maxvisit_init: f64,
    value_scale: f64,
    epsilon: f64,
    visit_cap: u64,
    rescale_values: bool,
) -> PyResult<Vec<f64>> {
    completed(
        value,
        &visits,
        &qvalues,
        &prior_probs,
        maxvisit_init,
        value_scale,
        epsilon,
        visit_cap,
        rescale_values,
    )
}

// Exposed for differential tests against math.fsum: the dispatching sum and
// the expansion fallback alone.
#[pyfunction]
fn _fsum(values: Vec<f64>) -> f64 {
    fsum(values.iter().copied())
}

#[pyfunction]
fn _expansion_sum(values: Vec<f64>) -> f64 {
    expansion_sum(values.iter().copied())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NodeStats>()?;
    m.add_function(wrap_pyfunction!(_fsum, m)?)?;
    m.add_function(wrap_pyfunction!(_expansion_sum, m)?)?;
    m.add_function(wrap_pyfunction!(gumbel_backup, m)?)?;
    m.add_function(wrap_pyfunction!(_gumbel_completed_q, m)?)?;
    Ok(())
}
