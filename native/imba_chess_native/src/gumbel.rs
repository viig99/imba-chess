//! Fused scalar Gumbel selectors. Tree ownership and RNG remain in Python.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// Expansion summation, including the final half-even correction. Ordinary
// compensated running sums do not reproduce math.fsum on cancellation/ties.
fn fsum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::with_capacity(32);
    for mut x in values {
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
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
    if let Some(&tail) = partials.last() {
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

fn completed(
    value: f64,
    visits: &[u64],
    qvalues: &[f64],
    probs: &[f64],
    maxvisit_init: f64,
    value_scale: f64,
    epsilon: f64,
) -> PyResult<Vec<f64>> {
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
    let mut values: Vec<f64> = qvalues
        .iter()
        .zip(visits)
        .map(|(q, n)| if *n != 0 { *q } else { mixed })
        .collect();
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scale = (maxvisit_init + *visits.iter().max().unwrap() as f64) * value_scale;
    let denominator = (high - low).max(epsilon);
    for q in &mut values {
        *q = scale * (*q - low) / denominator;
    }
    Ok(values)
}

fn validate_priors(priors: &[f64], n: usize) -> PyResult<()> {
    if priors.len() != n || priors.iter().any(|p| !p.is_finite()) {
        return Err(PyValueError::new_err("invalid prior logits"));
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn gumbel_interior_action(
    value: f64,
    priors: Vec<f64>,
    visits: Vec<u64>,
    qvalues: Vec<f64>,
    prior_probs: Vec<f64>,
    maxvisit_init: f64,
    value_scale: f64,
    epsilon: f64,
) -> PyResult<usize> {
    validate_priors(&priors, visits.len())?;
    let q = completed(
        value,
        &visits,
        &qvalues,
        &prior_probs,
        maxvisit_init,
        value_scale,
        epsilon,
    )?;
    let logits: Vec<f64> = priors.iter().zip(q).map(|(p, q)| p + q).collect();
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits.iter().map(|x| (x - maximum).exp()).collect();
    let total = fsum(weights.iter().copied());
    let denominator = (visits.iter().map(|n| *n as u128).sum::<u128>() + 1) as f64;
    let mut action = 0;
    let mut best = f64::NEG_INFINITY;
    for (i, (weight, n)) in weights.iter().zip(visits).enumerate() {
        let score = weight / total - n as f64 / denominator;
        if score > best {
            // Strict comparison preserves first-index ties.
            best = score;
            action = i;
        }
    }
    Ok(action)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn gumbel_root_action(
    value: f64,
    priors: Vec<f64>,
    visits: Vec<u64>,
    qvalues: Vec<f64>,
    prior_probs: Vec<f64>,
    maxvisit_init: f64,
    value_scale: f64,
    epsilon: f64,
    noise: Vec<f64>,
    eligible_visit: u64,
    max_prior: f64,
) -> PyResult<usize> {
    validate_priors(&priors, visits.len())?;
    if noise.len() != visits.len() || noise.iter().any(|x| !x.is_finite()) || !max_prior.is_finite()
    {
        return Err(PyValueError::new_err("invalid root noise"));
    }
    let q = completed(
        value,
        &visits,
        &qvalues,
        &prior_probs,
        maxvisit_init,
        value_scale,
        epsilon,
    )?;
    let mut action = None;
    let mut best = f64::NEG_INFINITY;
    for i in 0..visits.len() {
        if visits[i] == eligible_visit {
            let score = (-1e9f64).max(noise[i] + priors[i] - max_prior + q[i]);
            if score > best {
                best = score;
                action = Some(i);
            }
        }
    }
    action.ok_or_else(|| PyValueError::new_err("no eligible root action"))
}

// Exposed for direct numerical parity tests; selectors never round-trip Q arrays.
#[pyfunction]
fn _gumbel_completed_q(
    value: f64,
    visits: Vec<u64>,
    qvalues: Vec<f64>,
    prior_probs: Vec<f64>,
    maxvisit_init: f64,
    value_scale: f64,
    epsilon: f64,
) -> PyResult<Vec<f64>> {
    completed(
        value,
        &visits,
        &qvalues,
        &prior_probs,
        maxvisit_init,
        value_scale,
        epsilon,
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gumbel_interior_action, m)?)?;
    m.add_function(wrap_pyfunction!(gumbel_root_action, m)?)?;
    m.add_function(wrap_pyfunction!(_gumbel_completed_q, m)?)?;
    Ok(())
}
