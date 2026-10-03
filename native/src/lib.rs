//! Opt-in native backend prototype for `lisjong.hand_evaluation` (Issue #213).
//!
//! Plain Rust numeric shanten core and exact R5 search, with an optional
//! Python boundary (Issue #240). The default `python` feature preserves the
//! existing extension API; disabling it builds and tests without PyO3.
//!
//! The exact frontier artifact and helper tables still come from the Python
//! package. This is a dependency separation, not yet a standalone production
//! runtime or a public Rust API. Calculation semantics are unchanged.
//!
//! `python.rs` owns input validation, exception conversion, diagnostics and
//! module registration, including the existing batched discard entry.
//! `progression.rs` owns the Python-free sequential draw/discard DP.
//!
//! Integer range: frontier entry scores are 4-bit (0..=15) and at most four
//! groups are summed, so every intermediate score is in 0..=60 and the final
//! value is in -1..=8.  `i16` / `i32` cannot overflow for any input accepted by
//! the Python boundary (each count 0..=4, exactly 34 counts, `fixed_meld_count`
//! 0..=4).  Artifact dimensions are checked with `u64` arithmetic.

// The numeric core has no Python dependency. The existing wheel enables
// the binding by default; --no-default-features builds/tests only Rust.
mod progression;
#[cfg(feature = "python")]
mod python;

const MAGIC: &[u8; 8] = b"LISJSHT\x01";
const FORMAT_VERSION: u32 = 1;
const HEADER_SIZE: usize = 28; // struct "<8sIIIII"

const SUIT_KEY_SPACE: usize = 1_953_125; // 5**9
const HONOR_KEY_SPACE: usize = 78_125; // 5**7

pub(crate) const TILE_KIND_COUNT: usize = 34;
const MAX_FIXED_MELDS: i64 = 4;

const RESOURCE_STATE_COUNT: usize = 360; // 5 * 2 * 6 * 6
const SCORE_SHIFT: u16 = 4;
const SCORE_MASK: u16 = 0x0F;
const INVALID_STATE: u16 = 0xFFFF;
const UNREACHABLE_PENALTY: i8 = 127;
const STANDARD_SHANTEN_BASE: i32 = 8;

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

/// Reject pool entries whose resource state is not a real state.
///
/// `compute()` uses `packed >> SCORE_SHIFT` directly as an index into the
/// 360-entry score / stamp arrays and as the column of a 360-wide combine row.
/// A 12-bit state can reach 4095, so an unchecked value would index past the
/// arrays (a panic, which aborts the process in release) or, for 360..4095
/// values that still land inside the combine table, silently read another
/// row.  Every pool entry is checked here, once, at construction; together
/// with `parse_combine()` this makes every state the hot path can see
/// `< RESOURCE_STATE_COUNT`, so the hot path needs no per-entry check.
fn validate_pool_states(group: &FrontierGroup, label: &str) -> Result<(), String> {
    if group
        .pool
        .iter()
        .any(|&packed| (packed >> SCORE_SHIFT) as usize >= RESOURCE_STATE_COUNT)
    {
        return Err(format!(
            "{label} entry pool of the shanten table artifact references an unknown \
             resource state"
        ));
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
    validate_pool_states(&suit, "suit")?;
    validate_pool_states(&honor, "honor")?;
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

fn group_key(counts: &[u8]) -> usize {
    counts
        .iter()
        .fold(0usize, |key, &count| key * 5 + count as usize)
}

/// Immutable numeric shanten data shared with native searches.
///
/// It owns no Python object, so a reference to it (or an `Arc` clone) can cross
/// into a section where the GIL is released.
struct ShantenCore {
    suit: FrontierGroup,
    honor: FrontierGroup,
    combine: Vec<u16>,
    penalties: Vec<i8>,
}

impl ShantenCore {
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
}

impl ShantenCore {
    /// Same dispatch as `shanten._shanten_from_valid_counts()`; `total` must
    /// equal the sum of `counts` and be a valid concealed hand size.
    fn numeric_shanten(&self, counts: &[u8; TILE_KIND_COUNT], total: usize) -> Result<i32, String> {
        let fixed_meld_count = 4 - (total - 1) / 3;
        let mut shanten = self.compute(counts, fixed_meld_count)?;
        if total == 13 || total == 14 {
            shanten = shanten
                .min(seven_pairs_shanten(counts))
                .min(thirteen_orphans_shanten(counts));
        }
        Ok(shanten)
    }
}

pub(crate) const VALID_CONCEALED_TILE_COUNTS: [i64; 10] = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seven_pairs_counts_a_quad_as_one_pair() {
        let mut counts = [0; TILE_KIND_COUNT];
        counts[..7].fill(2);
        assert_eq!(seven_pairs_shanten(&counts), -1);
        counts[6] = 1;
        assert_eq!(seven_pairs_shanten(&counts), 0);
        counts[6] = 0;
        counts[0] = 4;
        assert_eq!(seven_pairs_shanten(&counts), 1);
    }

    #[test]
    fn thirteen_orphans_requires_terminal_and_honor_kinds_and_a_pair() {
        let mut counts = [0; TILE_KIND_COUNT];
        for index in TERMINAL_OR_HONOR_INDICES {
            counts[index] = 1;
        }
        assert_eq!(thirteen_orphans_shanten(&counts), 0);
        counts[0] = 2;
        assert_eq!(thirteen_orphans_shanten(&counts), -1);
        counts[33] = 0;
        counts[1] = 1;
        assert_eq!(thirteen_orphans_shanten(&counts), 0);
    }

    #[test]
    fn malformed_artifact_headers_are_rejected_without_python() {
        assert!(parse_artifact(&[]).is_err());
        let mut header = vec![0; HEADER_SIZE];
        assert!(parse_artifact(&header).is_err());
        header[..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        // Header is correct, but the declared dense indexes are missing.
        assert!(parse_artifact(&header).is_err());
    }

    #[test]
    fn combine_table_rejects_unknown_states_but_accepts_sentinel() {
        assert!(parse_combine(&[]).is_err());
        let mut bytes = vec![0xff; RESOURCE_STATE_COUNT * RESOURCE_STATE_COUNT * 2];
        assert!(
            parse_combine(&bytes)
                .unwrap()
                .iter()
                .all(|&v| v == INVALID_STATE)
        );
        bytes[..2].copy_from_slice(&(RESOURCE_STATE_COUNT as u16).to_le_bytes());
        assert!(parse_combine(&bytes).is_err());
    }

    #[test]
    fn frontier_spans_and_resource_states_are_checked() {
        let mut group = FrontierGroup {
            ids: vec![0],
            starts: vec![0],
            lengths: vec![1],
            pool: vec![0],
        };
        assert!(validate_spans(&group, "test").is_ok());
        assert!(validate_pool_states(&group, "test").is_ok());
        group.lengths[0] = 2;
        assert!(validate_spans(&group, "test").is_err());
        group.pool[0] = (RESOURCE_STATE_COUNT as u16) << SCORE_SHIFT;
        assert!(validate_pool_states(&group, "test").is_err());
        group.ids[0] = 1;
        assert!(group.entries(0, "test").is_err());
    }

    #[test]
    fn penalty_table_preserves_signed_values_and_requires_all_meld_counts() {
        assert!(parse_penalties(&[]).is_err());
        let mut bytes = vec![127; (MAX_FIXED_MELDS as usize + 1) * RESOURCE_STATE_COUNT];
        bytes[0] = 255;
        let values = parse_penalties(&bytes).unwrap();
        assert_eq!(values[0], -1);
        assert_eq!(values[1], UNREACHABLE_PENALTY);
    }
}
