//! Opt-in native backend prototype for `lisjong.hand_evaluation` (Issue #213).
//!
//! This crate ports exactly one private computation boundary: the numeric
//! shanten core of `lisjong.hand_evaluation.shanten` for trusted canonical
//! counts, i.e. the runtime frontier combine of
//! `_lookup_shanten.calculate_standard_shanten()` plus the closed-hand
//! seven-pairs / thirteen-orphans minimum of `_shanten_from_valid_counts()`.
//! It is not a new shanten semantic and it owns no data of its own:
//!
//! - the exact local frontier table is the same `_shanten_table.bin` payload
//!   that the Python backend reads (the Python side reads the package resource
//!   and passes the bytes in);
//! - the resource-state combine table and the per-`fixed_meld_count` penalty
//!   tables are the arrays already built by `_lookup_shanten` and passed in as
//!   bytes, so the seed / head / block-budget semantics stay single-sourced in
//!   Python.
//!
//! Only the lookup, the stable smallest-first group ordering, the max-score
//! combine, and the final penalty scan run here.  Every artifact integrity
//! failure the Python backend reports as `ShantenTableError` is reported with
//! the exception type passed in by the caller.  There is no fallback.
//!
//! Integer range: frontier entry scores are 4-bit (0..=15) and at most four
//! groups are summed, so every intermediate score is in 0..=60 and the final
//! value is in -1..=8.  `i16` / `i32` cannot overflow for any input accepted by
//! `read_counts()` (each count 0..=4, exactly 34 counts, `fixed_meld_count`
//! 0..=4).  Artifact dimensions are checked with `u64` arithmetic.

use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple, PyType};

const MAGIC: &[u8; 8] = b"LISJSHT\x01";
const FORMAT_VERSION: u32 = 1;
const HEADER_SIZE: usize = 28; // struct "<8sIIIII"

const SUIT_KEY_SPACE: usize = 1_953_125; // 5**9
const HONOR_KEY_SPACE: usize = 78_125; // 5**7

const TILE_KIND_COUNT: usize = 34;
const MAX_COPIES: i64 = 4;
const MAX_FIXED_MELDS: i64 = 4;

const RESOURCE_STATE_COUNT: usize = 360; // 5 * 2 * 6 * 6
const SCORE_SHIFT: u16 = 4;
const SCORE_MASK: u16 = 0x0F;
const INVALID_STATE: u16 = 0xFFFF;
const UNREACHABLE_PENALTY: i8 = 127;
const STANDARD_SHANTEN_BASE: i32 = 8;

/// Number of successful native shanten evaluations in this process.
///
/// Diagnostic only: tests use it to prove that the Rust path actually ran when
/// the Rust backend was explicitly selected.
static STANDARD_SHANTEN_CALLS: AtomicU64 = AtomicU64::new(0);

struct FrontierGroup {
    ids: Vec<u16>,
    starts: Vec<u32>,
    lengths: Vec<u8>,
    pool: Vec<u16>,
}

impl FrontierGroup {
    fn entries(&self, key: usize, label: &str) -> Result<&[u16], String> {
        let frontier_id = self.ids[key] as usize;
        if frontier_id >= self.starts.len() {
            return Err(format!(
                "{label} key index of the shanten table artifact references a \
                 frontier that does not exist"
            ));
        }
        let start = self.starts[frontier_id] as usize;
        let length = self.lengths[frontier_id] as usize;
        // Spans were validated against the pool at load time.
        Ok(&self.pool[start..start + length])
    }
}

struct Reader<'a> {
    payload: &'a [u8],
    offset: usize,
}

impl Reader<'_> {
    fn u16s(&mut self, count: usize) -> Vec<u16> {
        let end = self.offset + count * 2;
        let values = self.payload[self.offset..end]
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        self.offset = end;
        values
    }

    fn u32s(&mut self, count: usize) -> Vec<u32> {
        let end = self.offset + count * 4;
        let values = self.payload[self.offset..end]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        self.offset = end;
        values
    }

    fn u8s(&mut self, count: usize) -> Vec<u8> {
        let end = self.offset + count;
        let values = self.payload[self.offset..end].to_vec();
        self.offset = end;
        values
    }
}

fn header_u32(payload: &[u8], index: usize) -> u32 {
    let offset = 8 + index * 4;
    u32::from_le_bytes([
        payload[offset],
        payload[offset + 1],
        payload[offset + 2],
        payload[offset + 3],
    ])
}

fn validate_spans(group: &FrontierGroup, label: &str) -> Result<(), String> {
    if group.starts.is_empty() || group.lengths.len() != group.starts.len() {
        return Err(format!(
            "{label} frontier index of the shanten table artifact is empty or inconsistent"
        ));
    }
    let pool_length = group.pool.len() as u64;
    for (start, length) in group.starts.iter().zip(&group.lengths) {
        if *start as u64 + *length as u64 > pool_length {
            return Err(format!(
                "{label} frontier span of the shanten table artifact reaches past \
                 the end of its entry pool"
            ));
        }
    }
    Ok(())
}

fn parse_artifact(payload: &[u8]) -> Result<(FrontierGroup, FrontierGroup), String> {
    if payload.len() < HEADER_SIZE {
        return Err("shanten table artifact is truncated".to_owned());
    }
    if &payload[..8] != MAGIC {
        return Err("shanten table artifact has an unexpected magic".to_owned());
    }
    let version = header_u32(payload, 0);
    if version != FORMAT_VERSION {
        return Err(format!(
            "shanten table artifact format version is {version}, expected {FORMAT_VERSION}"
        ));
    }
    let suit_frontier_count = header_u32(payload, 1) as u64;
    let honor_frontier_count = header_u32(payload, 2) as u64;
    let suit_pool_entries = header_u32(payload, 3) as u64;
    let honor_pool_entries = header_u32(payload, 4) as u64;

    let expected = HEADER_SIZE as u64
        + (SUIT_KEY_SPACE + HONOR_KEY_SPACE) as u64 * 2
        + (suit_frontier_count + honor_frontier_count) * 5
        + (suit_pool_entries + honor_pool_entries) * 2;
    if payload.len() as u64 != expected {
        return Err(format!(
            "shanten table artifact size does not match its declared dimensions: \
             {} bytes, expected {expected}",
            payload.len()
        ));
    }

    let mut reader = Reader {
        payload,
        offset: HEADER_SIZE,
    };
    let suit_ids = reader.u16s(SUIT_KEY_SPACE);
    let honor_ids = reader.u16s(HONOR_KEY_SPACE);
    let suit_starts = reader.u32s(suit_frontier_count as usize);
    let suit_lengths = reader.u8s(suit_frontier_count as usize);
    let honor_starts = reader.u32s(honor_frontier_count as usize);
    let honor_lengths = reader.u8s(honor_frontier_count as usize);
    let suit_pool = reader.u16s(suit_pool_entries as usize);
    let honor_pool = reader.u16s(honor_pool_entries as usize);

    let suit = FrontierGroup {
        ids: suit_ids,
        starts: suit_starts,
        lengths: suit_lengths,
        pool: suit_pool,
    };
    let honor = FrontierGroup {
        ids: honor_ids,
        starts: honor_starts,
        lengths: honor_lengths,
        pool: honor_pool,
    };
    validate_spans(&suit, "suit")?;
    validate_spans(&honor, "honor")?;
    Ok((suit, honor))
}

fn parse_combine(bytes: &[u8]) -> Result<Vec<u16>, String> {
    if bytes.len() != RESOURCE_STATE_COUNT * RESOURCE_STATE_COUNT * 2 {
        return Err("resource-state combine table has an unexpected size".to_owned());
    }
    let values: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();
    if values
        .iter()
        .any(|&state| state != INVALID_STATE && state as usize >= RESOURCE_STATE_COUNT)
    {
        return Err("resource-state combine table references an unknown state".to_owned());
    }
    Ok(values)
}

fn parse_penalties(bytes: &[u8]) -> Result<Vec<i8>, String> {
    if bytes.len() != (MAX_FIXED_MELDS as usize + 1) * RESOURCE_STATE_COUNT {
        return Err("penalty tables have an unexpected size".to_owned());
    }
    Ok(bytes.iter().map(|&byte| byte as i8).collect())
}

fn read_count(item: &Bound<'_, PyAny>) -> PyResult<u8> {
    let value: i64 = item.extract()?;
    if !(0..=MAX_COPIES).contains(&value) {
        return Err(PyValueError::new_err(
            "counts must contain only values from 0 to 4",
        ));
    }
    Ok(value as u8)
}

/// Read a canonical 34-count sequence, rejecting values that would index the
/// dense key space out of range.  The Python backend trusts its caller for
/// this precondition; the native backend must check it for memory safety.
fn read_counts(counts: &Bound<'_, PyAny>) -> PyResult<[u8; TILE_KIND_COUNT]> {
    let mut values = [0u8; TILE_KIND_COUNT];
    let length_error = || PyValueError::new_err("counts must contain exactly 34 values");
    if let Ok(tuple) = counts.cast::<PyTuple>() {
        if tuple.len() != TILE_KIND_COUNT {
            return Err(length_error());
        }
        for (slot, item) in values.iter_mut().zip(tuple.iter()) {
            *slot = read_count(&item)?;
        }
    } else if let Ok(list) = counts.cast::<PyList>() {
        if list.len() != TILE_KIND_COUNT {
            return Err(length_error());
        }
        for (slot, item) in values.iter_mut().zip(list.iter()) {
            *slot = read_count(&item)?;
        }
    } else {
        if counts.len()? != TILE_KIND_COUNT {
            return Err(length_error());
        }
        for (index, slot) in values.iter_mut().enumerate() {
            *slot = read_count(&counts.get_item(index)?)?;
        }
    }
    Ok(values)
}

fn group_key(counts: &[u8]) -> usize {
    counts
        .iter()
        .fold(0usize, |key, &count| key * 5 + count as usize)
}

/// Exact standard-form shanten from the frontier table.
///
/// Mirrors `_lookup_shanten.calculate_standard_shanten()` step by step.
#[pyclass(frozen, module = "_lisjong_native")]
struct StandardShantenTable {
    suit: FrontierGroup,
    honor: FrontierGroup,
    combine: Vec<u16>,
    penalties: Vec<i8>,
    error_type: Py<PyType>,
}

impl StandardShantenTable {
    fn compute(
        &self,
        counts: &[u8; TILE_KIND_COUNT],
        fixed_meld_count: usize,
    ) -> Result<i32, String> {
        let mut groups: [&[u16]; 4] = [
            self.suit.entries(group_key(&counts[0..9]), "suit")?,
            self.suit.entries(group_key(&counts[9..18]), "suit")?,
            self.suit.entries(group_key(&counts[18..27]), "suit")?,
            self.honor.entries(group_key(&counts[27..34]), "honor")?,
        ];
        // Stable sort, same as Python `list.sort(key=length)`.  The combine is
        // associative, so the order only changes the cost, never the result.
        groups.sort_by_key(|entries| entries.len());

        let mut current = [-1i16; RESOURCE_STATE_COUNT];
        let mut next = [-1i16; RESOURCE_STATE_COUNT];
        let mut touched = [0u16; RESOURCE_STATE_COUNT];
        let mut next_touched = [0u16; RESOURCE_STATE_COUNT];
        let mut touched_count = 0usize;

        for &packed in groups[0] {
            let state = (packed >> SCORE_SHIFT) as usize;
            let score = (packed & SCORE_MASK) as i16;
            if current[state] < 0 {
                current[state] = score;
                touched[touched_count] = state as u16;
                touched_count += 1;
            } else if score > current[state] {
                current[state] = score;
            }
        }

        for entries in &groups[1..] {
            next.fill(-1);
            let mut next_count = 0usize;
            for &left_state in &touched[..touched_count] {
                let left_score = current[left_state as usize];
                let row = left_state as usize * RESOURCE_STATE_COUNT;
                for &packed in *entries {
                    let combined = self.combine[row + (packed >> SCORE_SHIFT) as usize];
                    if combined == INVALID_STATE {
                        continue;
                    }
                    let combined = combined as usize;
                    let score = left_score + (packed & SCORE_MASK) as i16;
                    if next[combined] < 0 {
                        next[combined] = score;
                        next_touched[next_count] = combined as u16;
                        next_count += 1;
                    } else if score > next[combined] {
                        next[combined] = score;
                    }
                }
            }
            std::mem::swap(&mut current, &mut next);
            std::mem::swap(&mut touched, &mut next_touched);
            touched_count = next_count;
        }

        if touched_count == 0 {
            return Err(
                "shanten table produced no reachable decomposition for the hand".to_owned(),
            );
        }

        let penalties = &self.penalties[fixed_meld_count * RESOURCE_STATE_COUNT
            ..(fixed_meld_count + 1) * RESOURCE_STATE_COUNT];
        let mut best: Option<i32> = None;
        for &state in &touched[..touched_count] {
            let penalty = penalties[state as usize];
            if penalty == UNREACHABLE_PENALTY {
                continue;
            }
            let value = current[state as usize] as i32 - penalty as i32;
            if best.is_none_or(|best| value > best) {
                best = Some(value);
            }
        }
        match best {
            Some(best) => Ok(STANDARD_SHANTEN_BASE - 2 * fixed_meld_count as i32 - best),
            None => {
                Err("shanten table produced no decomposition within the meld budget".to_owned())
            }
        }
    }

    fn table_error(&self, py: Python<'_>, message: String) -> PyErr {
        PyErr::from_type(self.error_type.bind(py).clone(), message)
    }
}

#[pymethods]
impl StandardShantenTable {
    /// Build the native view from the Python-owned artifact and tables.
    ///
    /// `payload` is the exact `_shanten_table.bin` content, `combine` the
    /// little-endian `array("H")` resource-state combine table, `penalties`
    /// the five concatenated `array("b")` penalty tables, and `error_type`
    /// the exception raised for artifact integrity failures.
    #[new]
    fn new(
        payload: &Bound<'_, PyBytes>,
        combine: &Bound<'_, PyBytes>,
        penalties: &Bound<'_, PyBytes>,
        error_type: Bound<'_, PyType>,
    ) -> PyResult<Self> {
        let raise = |message: String| PyErr::from_type(error_type.clone(), message);
        let (suit, honor) = parse_artifact(payload.as_bytes()).map_err(raise)?;
        let combine = parse_combine(combine.as_bytes()).map_err(raise)?;
        let penalties = parse_penalties(penalties.as_bytes()).map_err(raise)?;
        Ok(Self {
            suit,
            honor,
            combine,
            penalties,
            error_type: error_type.clone().unbind(),
        })
    }

    /// Return the standard-form shanten for trusted canonical counts.
    fn standard_shanten(
        &self,
        py: Python<'_>,
        counts: &Bound<'_, PyAny>,
        fixed_meld_count: i64,
    ) -> PyResult<i32> {
        let counts = read_counts(counts)?;
        if !(0..=MAX_FIXED_MELDS).contains(&fixed_meld_count) {
            return Err(PyValueError::new_err(
                "fixed_meld_count must be from 0 to 4",
            ));
        }
        let value = self
            .compute(&counts, fixed_meld_count as usize)
            .map_err(|message| self.table_error(py, message))?;
        STANDARD_SHANTEN_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(value)
    }

    /// Numeric shanten for trusted canonical counts.
    ///
    /// Mirrors `shanten._shanten_from_valid_counts()`: the standard form for
    /// every valid concealed size, and additionally the seven-pairs /
    /// thirteen-orphans minimum for closed 13 / 14-tile hands.
    /// `concealed_tile_count` must equal `sum(counts)` and be a valid concealed
    /// hand size.
    fn shanten_from_valid_counts(
        &self,
        py: Python<'_>,
        counts: &Bound<'_, PyAny>,
        concealed_tile_count: i64,
    ) -> PyResult<i32> {
        let counts = read_counts(counts)?;
        let total: i64 = counts.iter().map(|&count| count as i64).sum();
        if total != concealed_tile_count || !VALID_CONCEALED_TILE_COUNTS.contains(&total) {
            return Err(PyValueError::new_err(
                "concealed_tile_count must equal sum(counts) and be a valid concealed hand size",
            ));
        }
        let fixed_meld_count = (4 - (total - 1) / 3) as usize;
        let mut shanten = self
            .compute(&counts, fixed_meld_count)
            .map_err(|message| self.table_error(py, message))?;
        if total == 13 || total == 14 {
            shanten = shanten
                .min(seven_pairs_shanten(&counts))
                .min(thirteen_orphans_shanten(&counts));
        }
        STANDARD_SHANTEN_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(shanten)
    }
}

const VALID_CONCEALED_TILE_COUNTS: [i64; 10] = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14];
const TERMINAL_OR_HONOR_INDICES: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];

/// Same definition as `_python_shanten.calculate_seven_pairs_shanten()`.
fn seven_pairs_shanten(counts: &[u8; TILE_KIND_COUNT]) -> i32 {
    let pair_count = counts.iter().filter(|&&count| count >= 2).count() as i32;
    let kind_count = counts.iter().filter(|&&count| count >= 1).count() as i32;
    let mut shanten = 6 - pair_count;
    if kind_count < 7 {
        shanten += 7 - kind_count;
    }
    shanten
}

/// Same definition as `_python_shanten.calculate_thirteen_orphans_shanten()`.
fn thirteen_orphans_shanten(counts: &[u8; TILE_KIND_COUNT]) -> i32 {
    let kind_count = TERMINAL_OR_HONOR_INDICES
        .iter()
        .filter(|&&index| counts[index] >= 1)
        .count() as i32;
    let has_pair = TERMINAL_OR_HONOR_INDICES
        .iter()
        .any(|&index| counts[index] >= 2);
    13 - kind_count - i32::from(has_pair)
}

/// Number of successful native standard-shanten evaluations in this process.
#[pyfunction]
fn standard_shanten_call_count() -> u64 {
    STANDARD_SHANTEN_CALLS.load(Ordering::Relaxed)
}

#[pymodule]
fn _lisjong_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<StandardShantenTable>()?;
    module.add_function(wrap_pyfunction!(standard_shanten_call_count, module)?)?;
    Ok(())
}
