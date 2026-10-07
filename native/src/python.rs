//! Python boundary for the numeric core and R5 search.
//! Public extension contracts and validation are unchanged (Issue #240).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyInt, PyList, PyTuple, PyType};

use crate::scoring;
use crate::{
    MAX_FIXED_MELDS, ShantenCore, TILE_KIND_COUNT, VALID_CONCEALED_TILE_COUNTS, parse_artifact,
    parse_combine, parse_penalties, progression,
};
use progression::{Counts, SearchError};

const MAX_COPIES: i64 = 4;

/// Native entry-set version checked by `_shanten_backend` at import time.
///
/// 1 = `standard_shanten` / `shanten_from_valid_counts` (Issue #213; wheels
/// built before Issue #224 have no `API_VERSION` attribute),
/// 2 = additionally `evaluate_discards` (Issue #224),
/// 3 = additionally `evaluate_progression` (Issue #232).
const API_VERSION: u32 = 3;

/// Number of successful native shanten evaluations in this process.
///
/// Diagnostic only: tests use it to prove that the Rust path actually ran when
/// the Rust backend was explicitly selected.  It counts numeric-core
/// evaluations, not Python-to-Rust calls: one `evaluate_discards()` call adds
/// every evaluation it performed (only when the whole call succeeds).
static STANDARD_SHANTEN_CALLS: AtomicU64 = AtomicU64::new(0);

/// Number of successful `evaluate_discards()` calls (Python-to-Rust boundary
/// crossings of the batched entry) in this process.  Diagnostic only.
static DISCARD_EVALUATION_CALLS: AtomicU64 = AtomicU64::new(0);

/// Number of successful `evaluate_progression()` calls in this process.
/// Diagnostic only: tests use it to prove that the native R5 search ran.
static PROGRESSION_EVALUATION_CALLS: AtomicU64 = AtomicU64::new(0);

/// Concealed hand sizes from which a discard leaves a valid concealed size.
const DISCARD_HAND_SIZES: [usize; 5] = [2, 5, 8, 11, 14];

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

/// Exact standard-form shanten from the frontier table.
///
/// Mirrors `_lookup_shanten.calculate_standard_shanten()` step by step.
#[pyclass(frozen, module = "_lisjong_native")]
struct StandardShantenTable {
    core: Arc<ShantenCore>,
    error_type: Py<PyType>,
}

impl StandardShantenTable {
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
            core: Arc::new(ShantenCore {
                suit,
                honor,
                combine,
                penalties,
            }),
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
            .core
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
        let shanten = self
            .core
            .numeric_shanten(&counts, total as usize)
            .map_err(|message| self.table_error(py, message))?;
        STANDARD_SHANTEN_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(shanten)
    }

    /// Batched structural evaluation of discard candidates (Issue #224).
    ///
    /// `counts` is a canonical 34-count hand whose size allows a discard
    /// (2 / 5 / 8 / 11 / 14).  `discard_indexes` is an iterable of distinct
    /// canonical indexes (`int`, not `bool`), each held at least once.
    /// `improving_max_shanten` is `None` or an `int` (not `bool`).
    /// Returns `(shanten_after, improving_after)`, both in the order of
    /// `discard_indexes`:
    ///
    /// - `shanten_after[i]`: numeric shanten after discarding
    ///   `discard_indexes[i]`;
    /// - `improving_after[i]`: ascending tuple of the tile types that lower
    ///   `shanten_after[i]` when drawn after that discard, skipping types the
    ///   hand then holds four of (an evaluated empty set is `()`), or `None`
    ///   when it was not evaluated.  It is evaluated when
    ///   `improving_max_shanten` is `None` or
    ///   `shanten_after[i] <= improving_max_shanten`.
    ///
    /// Remaining inventory is not consulted.  Invalid input raises
    /// `TypeError` / `ValueError` before any evaluation; artifact failures
    /// raise the table's error type.  The input is not modified.
    #[pyo3(signature = (counts, discard_indexes, improving_max_shanten = None))]
    fn evaluate_discards<'py>(
        &self,
        py: Python<'py>,
        counts: &Bound<'py, PyAny>,
        discard_indexes: &Bound<'py, PyAny>,
        improving_max_shanten: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
        let mut counts = read_counts(counts)?;
        let total: usize = counts.iter().map(|&count| count as usize).sum();
        if !DISCARD_HAND_SIZES.contains(&total) {
            return Err(PyValueError::new_err(
                "counts must hold a concealed hand size that allows a discard \
                 (2, 5, 8, 11 or 14)",
            ));
        }
        let indexes = read_discard_indexes(discard_indexes, &counts)?;
        let improving_max_shanten = read_improving_max_shanten(improving_max_shanten)?;
        let table_error = |message: String| self.table_error(py, message);

        let mut evaluations = 0u64;
        let mut after_values = Vec::with_capacity(indexes.len());
        let mut improving_values = Vec::with_capacity(indexes.len());
        for &index in &indexes {
            counts[index] -= 1;
            evaluations += 1;
            let after = self
                .core
                .numeric_shanten(&counts, total - 1)
                .map_err(table_error)?;
            let evaluate_improving =
                improving_max_shanten.is_none_or(|maximum| after as i64 <= maximum);
            let improving = if evaluate_improving {
                let mut found = Vec::new();
                for drawn in 0..TILE_KIND_COUNT {
                    if counts[drawn] as i64 >= MAX_COPIES {
                        continue;
                    }
                    counts[drawn] += 1;
                    evaluations += 1;
                    let shanten = self.core.numeric_shanten(&counts, total);
                    counts[drawn] -= 1;
                    if shanten.map_err(table_error)? < after {
                        found.push(drawn as i64);
                    }
                }
                Some(found)
            } else {
                None
            };
            counts[index] += 1;
            after_values.push(after);
            improving_values.push(improving);
        }

        let improving_objects = improving_values
            .into_iter()
            .map(|improving| match improving {
                Some(found) => PyTuple::new(py, found).map(Bound::into_any),
                None => Ok(py.None().into_bound(py)),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let result = (
            PyTuple::new(py, after_values)?,
            PyTuple::new(py, improving_objects)?,
        );
        STANDARD_SHANTEN_CALLS.fetch_add(evaluations, Ordering::Relaxed);
        DISCARD_EVALUATION_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(result)
    }

    /// Exact terminal-shanten progression search for a batch of root hands
    /// (Issue #232).
    ///
    /// `root_hands` is a sequence of canonical 34-count post-discard hands,
    /// `remaining_counts` the canonical 34-count remaining inventory and
    /// `horizon` an `int` (not `bool`) from 1 to 3.  `policy_error_type` is the
    /// exception raised for semantic failures (terminal shanten outside the
    /// 0..=8 axis, `u64` range, a draw hand without a discard).
    ///
    /// Returns `(roots, counters)`: `roots[i]` is
    /// `(root_post_discard_shanten, terminal_shanten_counts)` in the order of
    /// `root_hands`, `counters` is `(visited_states, cache_hits, cache_misses,
    /// shanten_evaluations)` for the whole call.  All roots share one cache that
    /// lives only for this call.
    ///
    /// Input is fully converted to Rust-owned arrays and validated before the
    /// search; the search then runs with the GIL released and touches no Python
    /// object.  The GIL release does not make the search cancellable.
    #[pyo3(signature = (root_hands, remaining_counts, horizon, policy_error_type))]
    fn evaluate_progression<'py>(
        &self,
        py: Python<'py>,
        root_hands: &Bound<'py, PyAny>,
        remaining_counts: &Bound<'py, PyAny>,
        horizon: &Bound<'py, PyAny>,
        policy_error_type: &Bound<'py, PyType>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
        if horizon.is_instance_of::<PyBool>() || !horizon.is_instance_of::<PyInt>() {
            return Err(PyTypeError::new_err("horizon must be an int"));
        }
        let horizon = match horizon.extract::<i64>() {
            Ok(value) if (1..=3).contains(&value) => value as u8,
            _ => return Err(PyValueError::new_err("horizon must be from 1 to 3")),
        };
        let remaining = read_counts(remaining_counts)?;
        let remaining_total: u64 = remaining.iter().map(|&count| u64::from(count)).sum();
        if remaining_total < u64::from(horizon) {
            return Err(PyValueError::new_err(
                "remaining_counts must hold at least horizon tiles",
            ));
        }
        let mut roots: Vec<Counts> = Vec::new();
        for item in root_hands.try_iter()? {
            let hand = read_counts(&item?)?;
            let total: usize = hand.iter().map(|&count| usize::from(count)).sum();
            if !VALID_CONCEALED_TILE_COUNTS.contains(&(total as i64)) {
                return Err(PyValueError::new_err(
                    "root hands must hold a valid concealed hand size",
                ));
            }
            roots.push(hand);
        }

        // Rust-owned inputs and the shared immutable table only: no Python
        // object or borrow enters the detached section.
        let core = Arc::clone(&self.core);
        let outcome =
            py.detach(move || progression::search_roots(&core, &roots, &remaining, horizon));

        let output = outcome.map_err(|error| match error {
            SearchError::Table(message) => self.table_error(py, message),
            SearchError::Policy(message) => PyErr::from_type(policy_error_type.clone(), message),
            SearchError::Value(message) => PyValueError::new_err(message),
        })?;

        let root_objects = output
            .roots
            .iter()
            .map(|root| {
                let distribution = PyTuple::new(py, root.distribution)?;
                PyTuple::new(
                    py,
                    [
                        root.root_post_discard_shanten.into_pyobject(py)?.into_any(),
                        distribution.into_any(),
                    ],
                )
                .map(Bound::into_any)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let counters = output.counters;
        let result = (
            PyTuple::new(py, root_objects)?,
            PyTuple::new(
                py,
                [
                    counters.visited_states,
                    counters.cache_hits,
                    counters.cache_misses,
                    counters.shanten_evaluations,
                ],
            )?,
        );
        STANDARD_SHANTEN_CALLS.fetch_add(counters.shanten_evaluations, Ordering::Relaxed);
        PROGRESSION_EVALUATION_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(result)
    }
}

/// Read the optional improving threshold (`None` = evaluate every candidate).
fn read_improving_max_shanten(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<i64>> {
    let Some(value) = value.filter(|value| !value.is_none()) else {
        return Ok(None);
    };
    if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(
            "improving_max_shanten must be None or an int",
        ));
    }
    value
        .extract::<i64>()
        .map(Some)
        .map_err(|_| PyValueError::new_err("improving_max_shanten is out of range"))
}

/// Read distinct canonical discard indexes, each held in `counts`.
fn read_discard_indexes(
    indexes: &Bound<'_, PyAny>,
    counts: &[u8; TILE_KIND_COUNT],
) -> PyResult<Vec<usize>> {
    let mut seen = [false; TILE_KIND_COUNT];
    let mut values = Vec::new();
    for item in indexes.try_iter()? {
        let item = item?;
        if item.is_instance_of::<PyBool>() || !item.is_instance_of::<PyInt>() {
            return Err(PyTypeError::new_err(
                "discard_indexes must contain only int values",
            ));
        }
        let index = match item.extract::<i64>() {
            Ok(value) if (0..TILE_KIND_COUNT as i64).contains(&value) => value as usize,
            _ => {
                return Err(PyValueError::new_err(
                    "discard_indexes must contain only values from 0 to 33",
                ));
            }
        };
        if seen[index] {
            return Err(PyValueError::new_err(
                "discard_indexes must not contain duplicates",
            ));
        }
        seen[index] = true;
        if counts[index] == 0 {
            return Err(PyValueError::new_err(
                "discard_indexes must reference tile types held in counts",
            ));
        }
        values.push(index);
    }
    Ok(values)
}

/// Number of successful native standard-shanten evaluations in this process.
#[pyfunction]
fn standard_shanten_call_count() -> u64 {
    STANDARD_SHANTEN_CALLS.load(Ordering::Relaxed)
}

/// Number of successful `evaluate_discards()` calls in this process.
#[pyfunction]
fn discard_evaluation_call_count() -> u64 {
    DISCARD_EVALUATION_CALLS.load(Ordering::Relaxed)
}

/// Number of successful `evaluate_progression()` calls in this process.
#[pyfunction]
fn progression_evaluation_call_count() -> u64 {
    PROGRESSION_EVALUATION_CALLS.load(Ordering::Relaxed)
}

/// Full lisjong commit this extension was built from (Issue #216).
///
/// The extension shares `_shanten_table.bin` and the helper tables with the
/// Python package and mirrors its special-hand dispatch, so a wheel is only
/// validated together with the lisjong revision it was built from.  The wheel
/// CI job sets `LISJONG_NATIVE_SOURCE_REVISION`; a local source build without
/// it reports `"unknown"`.  Consumers record this value next to the wheel
/// SHA-256 and compare it with their pinned lisjong revision.
const SOURCE_REVISION: &str = match option_env!("LISJONG_NATIVE_SOURCE_REVISION") {
    Some(revision) => revision,
    None => "unknown",
};

/// Version of the scoring entry `evaluate_win` (Issue #263).  Checked by
/// `lisjong.hand_evaluation.scoring` independently of the shanten
/// `API_VERSION`, so the shanten backend contract is unchanged.
const SCORING_API_VERSION: u32 = 1;

type TileTuple = (u8, bool);

fn scoring_tile((kind, red): TileTuple) -> scoring::TileInput {
    scoring::TileInput { kind, red }
}

fn scoring_tiles(tiles: Vec<TileTuple>) -> Vec<scoring::TileInput> {
    tiles.into_iter().map(scoring_tile).collect()
}

fn unknown(field: &str, value: &str) -> PyErr {
    PyValueError::new_err(format!("unknown {field}: {value:?}"))
}

/// Score one concrete complete hand (Issue #263).
///
/// Arguments are primitive conversions of the `lisjong.hand_evaluation.scoring`
/// dataclasses: tiles are `(kind_34, is_red)` tuples, enums are their string
/// values, `ura_dora` is a tile list or `None` for the explicit exclusion
/// mode.  Returns `("not_complete", None)`, `("no_yaku", None)` or
/// `("scored", dict)`.  Invalid input raises `ValueError`.
#[pyfunction]
#[pyo3(signature = (
    concealed, melds, winning_tile, method, seat_wind, prevailing_wind, riichi,
    ippatsu, situation, dora_indicators, ura_dora, rules
))]
#[allow(clippy::too_many_arguments)]
fn evaluate_win<'py>(
    py: Python<'py>,
    concealed: Vec<TileTuple>,
    melds: Vec<(String, Vec<TileTuple>)>,
    winning_tile: TileTuple,
    method: &str,
    seat_wind: u8,
    prevailing_wind: u8,
    riichi: &str,
    ippatsu: bool,
    situation: &str,
    dora_indicators: Vec<TileTuple>,
    ura_dora: Option<Vec<TileTuple>>,
    rules: (bool, bool, bool, bool, bool, Vec<String>, u8),
) -> PyResult<(&'static str, Option<Bound<'py, PyDict>>)> {
    let melds = melds
        .into_iter()
        .map(|(kind, tiles)| {
            let kind = match kind.as_str() {
                "chi" => scoring::MeldKind::Chi,
                "pon" => scoring::MeldKind::Pon,
                "daiminkan" => scoring::MeldKind::Daiminkan,
                "ankan" => scoring::MeldKind::Ankan,
                "kakan" => scoring::MeldKind::Kakan,
                other => return Err(unknown("meld kind", other)),
            };
            Ok(scoring::MeldInput {
                kind,
                tiles: scoring_tiles(tiles),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let method = match method {
        "ron" => scoring::WinMethod::Ron,
        "tsumo" => scoring::WinMethod::Tsumo,
        other => return Err(unknown("win method", other)),
    };
    let riichi = match riichi {
        "none" => scoring::RiichiStatus::None,
        "riichi" => scoring::RiichiStatus::Riichi,
        "double_riichi" => scoring::RiichiStatus::DoubleRiichi,
        other => return Err(unknown("riichi status", other)),
    };
    let situation = match situation {
        "normal" => scoring::WinSituation::Normal,
        "haitei" => scoring::WinSituation::Haitei,
        "houtei" => scoring::WinSituation::Houtei,
        "rinshan" => scoring::WinSituation::Rinshan,
        "chankan" => scoring::WinSituation::Chankan,
        "tenhou" => scoring::WinSituation::Tenhou,
        "chiihou" => scoring::WinSituation::Chiihou,
        other => return Err(unknown("win situation", other)),
    };
    let (kuitan, red, rounded, counted, multiple, double_variants, double_wind_fu) = rules;
    let double_yakuman_variants = double_variants
        .iter()
        .map(|name| scoring::Yaku::from_name(name).ok_or_else(|| unknown("yaku", name)))
        .collect::<PyResult<Vec<_>>>()?;
    let input = scoring::WinInput {
        concealed: scoring_tiles(concealed),
        melds,
        winning_tile: scoring_tile(winning_tile),
        method,
        seat_wind,
        prevailing_wind,
        riichi,
        ippatsu,
        situation,
        dora_indicators: scoring_tiles(dora_indicators),
        ura_dora: match ura_dora {
            Some(tiles) => scoring::UraDora::Indicators(scoring_tiles(tiles)),
            None => scoring::UraDora::Excluded,
        },
    };
    let rules = scoring::Rules {
        kuitan_enabled: kuitan,
        red_dora_enabled: red,
        rounded_mangan_enabled: rounded,
        counted_yakuman_enabled: counted,
        multiple_yakuman_enabled: multiple,
        double_yakuman_variants,
        double_wind_pair_fu: double_wind_fu,
    };
    let score = match scoring::evaluate(&input, &rules).map_err(PyValueError::new_err)? {
        scoring::Evaluation::NotComplete => return Ok(("not_complete", None)),
        scoring::Evaluation::CompleteWithoutYaku => return Ok(("no_yaku", None)),
        scoring::Evaluation::Scored(score) => score,
    };
    let result = PyDict::new(py);
    result.set_item("shape", score.shape.name())?;
    result.set_item("wait", score.wait.name())?;
    result.set_item("pair", score.pair)?;
    let groups: Vec<(&str, u8, bool, bool)> = score
        .groups
        .iter()
        .map(|group| {
            (
                group.kind.name(),
                group.tile,
                group.is_open,
                group.completed_by_ron,
            )
        })
        .collect();
    result.set_item("groups", groups)?;
    result.set_item("completed_group", score.completed_group)?;
    let yaku: Vec<(&str, u8, u8)> = score
        .yaku
        .iter()
        .map(|value| (value.yaku.name(), value.han, value.yakuman_units))
        .collect();
    result.set_item("yaku", yaku)?;
    result.set_item(
        "dora",
        score.dora.map(|dora| (dora.dora, dora.red, dora.ura)),
    )?;
    result.set_item("ura_dora_excluded", score.ura_dora_excluded)?;
    result.set_item("yaku_han", score.yaku_han)?;
    result.set_item("han", score.han)?;
    result.set_item("fu", score.fu)?;
    let fu_components: Vec<(&str, u32)> = score
        .fu_components
        .iter()
        .map(|(reason, fu)| (reason.name(), *fu))
        .collect();
    result.set_item("fu_components", fu_components)?;
    result.set_item("yakuman_units", score.yakuman_units)?;
    result.set_item(
        "method",
        match score.method {
            scoring::WinMethod::Ron => "ron",
            scoring::WinMethod::Tsumo => "tsumo",
        },
    )?;
    result.set_item("is_dealer", score.is_dealer)?;
    result.set_item("base_points", score.base_points)?;
    result.set_item("limit", score.limit.name())?;
    result.set_item("ron_payment", score.ron_payment)?;
    result.set_item("tsumo_dealer_payment", score.tsumo_dealer_payment)?;
    result.set_item("tsumo_non_dealer_payment", score.tsumo_non_dealer_payment)?;
    result.set_item("winner_points", score.winner_points)?;
    Ok(("scored", Some(result)))
}

#[pymodule]
fn _lisjong_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("SOURCE_REVISION", SOURCE_REVISION)?;
    module.add("API_VERSION", API_VERSION)?;
    module.add("SCORING_API_VERSION", SCORING_API_VERSION)?;
    module.add_function(wrap_pyfunction!(evaluate_win, module)?)?;
    module.add_class::<StandardShantenTable>()?;
    module.add_function(wrap_pyfunction!(standard_shanten_call_count, module)?)?;
    module.add_function(wrap_pyfunction!(discard_evaluation_call_count, module)?)?;
    module.add_function(wrap_pyfunction!(progression_evaluation_call_count, module)?)?;
    Ok(())
}
