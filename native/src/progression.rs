//! Exact terminal-shanten progression search (Issue #232).
//!
//! Native port of `_TerminalShantenProgressionEvaluator` (R5, Issue #169): the
//! sequential draw / discard DP over `horizon` self-draws whose value is the
//! exact integer distribution of terminal structural shanten.  The Python
//! evaluator stays the oracle; this module mirrors it step by step so that the
//! distributions, masses and the four instrumentation counters match exactly:
//!
//! - children of a draw hand are ordered by `(shanten, hand_counts)`;
//! - the best child is replaced only on a strictly smaller mass;
//! - a child is skipped once its lower bound reaches the provisional best mass
//!   (`>=`);
//! - the cache is owned by one search call and is never carried across calls.
//!
//! Everything here is plain Rust data: no Python object crosses into the
//! search, so the caller can release the GIL around [`search_roots`].  Releasing
//! the GIL does not make the search cancellable.
//!
//! The crate builds with `panic = "abort"` and without release overflow checks,
//! so nothing below may panic or wrap for input-derived values: distribution
//! counts, masses and denominators are `u64` with checked arithmetic, and every
//! abnormal state is returned as a [`SearchError`].

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::{ShantenCore, TILE_KIND_COUNT, VALID_CONCEALED_TILE_COUNTS};

/// Terminal structural shanten axis (0..=8), same as
/// `TERMINAL_SHANTEN_AXIS` in the Python module.
pub(crate) const TERMINAL_SHANTEN_AXIS: usize = 9;

pub(crate) type Counts = [u8; TILE_KIND_COUNT];
pub(crate) type Distribution = [u64; TERMINAL_SHANTEN_AXIS];

/// Why a search stopped.  The caller maps each kind to a Python exception after
/// the GIL is re-acquired.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SearchError {
    /// Shanten table artifact failure (`ShantenTableError`).
    Table(String),
    /// Semantic failure the Python evaluator reports as
    /// `TerminalShantenProgressionPolicyError` (axis range, arithmetic range,
    /// a draw hand without a discard).
    Policy(String),
    /// Input that violates the evaluator's precondition (`ValueError`).
    Value(String),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Counters {
    pub visited_states: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub shanten_evaluations: u64,
}

pub(crate) struct RootResult {
    pub root_post_discard_shanten: i32,
    pub distribution: Distribution,
}

pub(crate) struct SearchOutput {
    pub roots: Vec<RootResult>,
    pub counters: Counters,
}

/// Multiply-rotate hasher (the Fx scheme) for the search caches.  The keys are
/// short, trusted count arrays, so SipHash's DoS resistance buys nothing and
/// costs a measurable share of the search.  The hasher only speeds up lookups:
/// it never changes which states exist, so results and counters are unaffected.
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

impl FxHasher {
    const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(Self::SEED);
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(word));
        }
    }

    fn write_u8(&mut self, value: u8) {
        self.add(u64::from(value));
    }

    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

#[derive(PartialEq, Eq, Hash)]
struct StateKey {
    hand: Counts,
    remaining: Counts,
    depth: u8,
}

fn policy_error(message: &str) -> SearchError {
    SearchError::Policy(message.to_owned())
}

fn arithmetic_error() -> SearchError {
    policy_error("terminal shanten progression arithmetic left the u64 range")
}

/// `F(count, length) = count * (count - 1) * ... * (count - length + 1)`.
fn falling_factorial(count: u64, length: u8) -> Result<u64, SearchError> {
    let mut value = 1u64;
    for step in 0..u64::from(length) {
        let factor = count.checked_sub(step).ok_or_else(arithmetic_error)?;
        value = value.checked_mul(factor).ok_or_else(arithmetic_error)?;
    }
    Ok(value)
}

/// `Σ index * count`, the exact integer terminal shanten mass.
fn distribution_mass(distribution: &Distribution) -> Result<u64, SearchError> {
    let mut mass = 0u64;
    for (index, &count) in distribution.iter().enumerate() {
        let term = (index as u64)
            .checked_mul(count)
            .ok_or_else(arithmetic_error)?;
        mass = mass.checked_add(term).ok_or_else(arithmetic_error)?;
    }
    Ok(mass)
}

/// `_require_supported_terminal_shanten()`: the distribution axis index.
fn axis_index(shanten: i32) -> Result<usize, SearchError> {
    if !(0..TERMINAL_SHANTEN_AXIS as i32).contains(&shanten) {
        return Err(SearchError::Policy(format!(
            "terminal structural shanten is outside the supported axis: {shanten}"
        )));
    }
    Ok(shanten as usize)
}

fn unit_distribution(shanten: i32) -> Result<Distribution, SearchError> {
    let mut counts = [0u64; TERMINAL_SHANTEN_AXIS];
    counts[axis_index(shanten)?] = 1;
    Ok(counts)
}

struct Search<'a> {
    core: &'a ShantenCore,
    shanten_cache: FastMap<Counts, i32>,
    distribution_cache: FastMap<StateKey, Distribution>,
    counters: Counters,
}

impl Search<'_> {
    fn shanten(&mut self, hand: &Counts) -> Result<i32, SearchError> {
        if let Some(&cached) = self.shanten_cache.get(hand) {
            return Ok(cached);
        }
        self.counters.shanten_evaluations += 1;
        let total: usize = hand.iter().map(|&count| usize::from(count)).sum();
        if !VALID_CONCEALED_TILE_COUNTS.contains(&(total as i64)) {
            return Err(SearchError::Value(format!(
                "a hand of {total} concealed tiles is not a valid concealed hand size"
            )));
        }
        let value = self
            .core
            .numeric_shanten(hand, total)
            .map_err(SearchError::Table)?;
        self.shanten_cache.insert(*hand, value);
        Ok(value)
    }

    /// `min_y shanten(Y - y) == max(0, shanten(Y))`.
    fn best_post_discard_shanten(&mut self, draw_hand: &Counts) -> Result<i32, SearchError> {
        Ok(self.shanten(draw_hand)?.max(0))
    }

    fn distribution(
        &mut self,
        hand: &Counts,
        remaining: &Counts,
        depth: u8,
        remaining_total: u64,
    ) -> Result<Distribution, SearchError> {
        if depth == 0 {
            return unit_distribution(self.shanten(hand)?);
        }
        let key = StateKey {
            hand: *hand,
            remaining: *remaining,
            depth,
        };
        if let Some(&cached) = self.distribution_cache.get(&key) {
            self.counters.cache_hits += 1;
            return Ok(cached);
        }
        self.counters.cache_misses += 1;
        let distribution = self.search(hand, remaining, depth, remaining_total)?;
        self.distribution_cache.insert(key, distribution);
        Ok(distribution)
    }

    fn search(
        &mut self,
        hand: &Counts,
        remaining: &Counts,
        depth: u8,
        remaining_total: u64,
    ) -> Result<Distribution, SearchError> {
        self.counters.visited_states += 1;
        let current_shanten = self.shanten(hand)?;
        let mut counts = [0u64; TERMINAL_SHANTEN_AXIS];
        if current_shanten == 0 {
            counts[0] = falling_factorial(remaining_total, depth)?;
            return Ok(counts);
        }

        if depth == 1 {
            for drawn in 0..TILE_KIND_COUNT {
                let available = remaining[drawn];
                if available == 0 {
                    continue;
                }
                let draw_hand = draw_tile(hand, drawn)?;
                let terminal = self.best_post_discard_shanten(&draw_hand)?;
                let index = axis_index(terminal)?;
                counts[index] = counts[index]
                    .checked_add(u64::from(available))
                    .ok_or_else(arithmetic_error)?;
            }
            return Ok(counts);
        }

        let child_depth = depth - 1;
        let child_remaining_total = remaining_total
            .checked_sub(1)
            .ok_or_else(arithmetic_error)?;
        let child_denominator = falling_factorial(child_remaining_total, child_depth)?;
        for drawn in 0..TILE_KIND_COUNT {
            let available = remaining[drawn];
            if available == 0 {
                continue;
            }
            let draw_hand = draw_tile(hand, drawn)?;
            let mut next_remaining = *remaining;
            next_remaining[drawn] -= 1;
            let best = self.best_discard_distribution(
                &draw_hand,
                &next_remaining,
                child_depth,
                child_remaining_total,
                child_denominator,
            )?;
            for (index, &value) in best.iter().enumerate() {
                if value != 0 {
                    let term = u64::from(available)
                        .checked_mul(value)
                        .ok_or_else(arithmetic_error)?;
                    counts[index] = counts[index]
                        .checked_add(term)
                        .ok_or_else(arithmetic_error)?;
                }
            }
        }
        Ok(counts)
    }

    fn best_discard_distribution(
        &mut self,
        draw_hand: &Counts,
        remaining: &Counts,
        depth: u8,
        remaining_total: u64,
        denominator: u64,
    ) -> Result<Distribution, SearchError> {
        let mut children: Vec<(i32, Counts)> = Vec::with_capacity(TILE_KIND_COUNT);
        for discarded in 0..TILE_KIND_COUNT {
            if draw_hand[discarded] == 0 {
                continue;
            }
            let mut child = *draw_hand;
            child[discarded] -= 1;
            let child_shanten = self.shanten(&child)?;
            children.push((child_shanten, child));
        }
        // `(shanten, hand_counts)` lexicographic order, as Python's
        // `children.sort()` on `(int, tuple[int, ...])`.
        children.sort();

        let mut best: Option<(Distribution, u64)> = None;
        for (child_shanten, child) in &children {
            let gap = (i64::from(*child_shanten) - i64::from(depth)).max(0) as u64;
            let lower_bound = gap.checked_mul(denominator).ok_or_else(arithmetic_error)?;
            if let Some((_, best_mass)) = &best
                && lower_bound >= *best_mass
            {
                break;
            }
            let distribution = self.distribution(child, remaining, depth, remaining_total)?;
            let mass = distribution_mass(&distribution)?;
            if best.as_ref().is_none_or(|(_, best_mass)| mass < *best_mass) {
                best = Some((distribution, mass));
            }
        }
        best.map(|(distribution, _)| distribution).ok_or_else(|| {
            policy_error("a draw hand must have at least one structural discard candidate")
        })
    }
}

/// `hand + one(drawn)`; a fifth copy of a base tile is a precondition failure.
fn draw_tile(hand: &Counts, drawn: usize) -> Result<Counts, SearchError> {
    if hand[drawn] >= 4 {
        return Err(SearchError::Value(
            "a hand and its remaining counts must not hold more than 4 copies of a tile".to_owned(),
        ));
    }
    let mut draw_hand = *hand;
    draw_hand[drawn] += 1;
    Ok(draw_hand)
}

/// Evaluate every root (post-discard) hand with one shared cache.
///
/// Mirrors `_evaluate_progression_candidates()` on the Python evaluator: for
/// each root in order, the terminal distribution, then the root shanten.
pub(crate) fn search_roots(
    core: &ShantenCore,
    roots: &[Counts],
    remaining: &Counts,
    horizon: u8,
) -> Result<SearchOutput, SearchError> {
    let remaining_total: u64 = remaining.iter().map(|&count| u64::from(count)).sum();
    let mut search = Search {
        core,
        shanten_cache: FastMap::default(),
        distribution_cache: FastMap::default(),
        counters: Counters::default(),
    };
    let mut results = Vec::with_capacity(roots.len());
    for root in roots {
        let distribution = search.distribution(root, remaining, horizon, remaining_total)?;
        let root_post_discard_shanten = search.shanten(root)?;
        results.push(RootResult {
            root_post_discard_shanten,
            distribution,
        });
    }
    Ok(SearchOutput {
        roots: results,
        counters: search.counters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrontierGroup;

    fn empty_group() -> FrontierGroup {
        FrontierGroup {
            ids: Vec::new(),
            starts: Vec::new(),
            lengths: Vec::new(),
            pool: Vec::new(),
        }
    }

    /// A table that must never be consulted: the failure paths below are
    /// reached before any shanten evaluation.
    fn unused_core() -> ShantenCore {
        ShantenCore {
            suit: empty_group(),
            honor: empty_group(),
            combine: Vec::new(),
            penalties: Vec::new(),
        }
    }

    #[test]
    fn falling_factorial_reports_overflow_instead_of_wrapping() {
        assert_eq!(falling_factorial(10, 3), Ok(720));
        assert_eq!(falling_factorial(5, 0), Ok(1));
        assert!(matches!(
            falling_factorial(u64::MAX, 2),
            Err(SearchError::Policy(_))
        ));
        assert!(matches!(
            falling_factorial(1, 3),
            Err(SearchError::Policy(_))
        ));
    }

    #[test]
    fn distribution_mass_reports_overflow_instead_of_wrapping() {
        let mut distribution = [0u64; TERMINAL_SHANTEN_AXIS];
        distribution[2] = 5;
        distribution[3] = 7;
        assert_eq!(distribution_mass(&distribution), Ok(31));
        distribution[8] = u64::MAX;
        assert!(matches!(
            distribution_mass(&distribution),
            Err(SearchError::Policy(_))
        ));
        let mut sum_overflow = [0u64; TERMINAL_SHANTEN_AXIS];
        sum_overflow[1] = u64::MAX;
        sum_overflow[2] = u64::MAX / 2;
        assert!(matches!(
            distribution_mass(&sum_overflow),
            Err(SearchError::Policy(_))
        ));
    }

    #[test]
    fn terminal_shanten_outside_the_axis_is_an_error() {
        assert_eq!(axis_index(0), Ok(0));
        assert_eq!(axis_index(8), Ok(8));
        for value in [-1, 9, i32::MIN, i32::MAX] {
            assert!(matches!(axis_index(value), Err(SearchError::Policy(_))));
            assert!(matches!(
                unit_distribution(value),
                Err(SearchError::Policy(_))
            ));
        }
    }

    #[test]
    fn a_draw_hand_without_a_discard_candidate_is_an_error() {
        let core = unused_core();
        let mut search = Search {
            core: &core,
            shanten_cache: FastMap::default(),
            distribution_cache: FastMap::default(),
            counters: Counters::default(),
        };
        let empty = [0u8; TILE_KIND_COUNT];
        let result = search.best_discard_distribution(&empty, &empty, 1, 1, 1);
        assert!(matches!(result, Err(SearchError::Policy(_))));
    }

    #[test]
    fn a_fifth_copy_of_a_tile_is_an_error() {
        let mut hand = [0u8; TILE_KIND_COUNT];
        hand[3] = 4;
        assert!(matches!(draw_tile(&hand, 3), Err(SearchError::Value(_))));
        assert!(draw_tile(&hand, 4).is_ok());
    }

    #[test]
    fn an_invalid_hand_size_is_an_error_before_any_table_access() {
        let core = unused_core();
        let mut search = Search {
            core: &core,
            shanten_cache: FastMap::default(),
            distribution_cache: FastMap::default(),
            counters: Counters::default(),
        };
        let mut hand = [0u8; TILE_KIND_COUNT];
        hand[0] = 3;
        hand[1] = 3;
        assert!(matches!(search.shanten(&hand), Err(SearchError::Value(_))));
    }
}
