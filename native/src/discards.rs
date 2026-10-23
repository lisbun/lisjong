//! Python-free batched structural evaluation of discard candidates
//! (Issue #224; moved out of the binding in Issue #333).
//!
//! For each discard candidate this enumerates the post-discard hand and every
//! virtual draw after it, evaluating each with the numeric shanten core.  It
//! holds no discard strategy (candidate selection, acceptance counting or
//! evaluation order).  Input validation, exception conversion, Python result
//! objects and diagnostic counters stay in `python.rs`.

use crate::{ShantenCore, TILE_KIND_COUNT};

type Counts = [u8; TILE_KIND_COUNT];

const MAX_COPIES: u8 = 4;

/// Concealed hand sizes from which a discard leaves a valid concealed size.
pub(crate) const DISCARD_HAND_SIZES: [usize; 5] = [2, 5, 8, 11, 14];

/// Result of one successful batched evaluation, in candidate order.
#[derive(Debug, PartialEq)]
pub(crate) struct DiscardEvaluation {
    /// Numeric shanten after discarding each candidate.
    pub(crate) shanten_after: Vec<i32>,
    /// Ascending tile types that lower `shanten_after[i]` when drawn after
    /// that discard (`Some(vec![])` when evaluated and empty), or `None` when
    /// the candidate was not evaluated.
    pub(crate) improving_after: Vec<Option<Vec<usize>>>,
    /// Number of numeric shanten evaluations performed.
    pub(crate) shanten_evaluations: u64,
}

/// Evaluate discard candidates with the numeric shanten core.
///
/// Preconditions (checked by the caller before any evaluation):
///
/// - every value of `counts` is from 0 to 4 and their sum is one of
///   `DISCARD_HAND_SIZES`;
/// - `discard_indexes` are distinct, below `TILE_KIND_COUNT`, and each is held
///   at least once in `counts`.
///
/// Improving tiles are evaluated when `improving_max_shanten` is `None` or the
/// post-discard shanten is `<=` it.  Drawn types the post-discard hand already
/// holds four of are skipped; remaining inventory is not consulted.  The first
/// table error is returned as is, without a partial result.  `counts` is not
/// modified.
pub(crate) fn evaluate_discards(
    core: &ShantenCore,
    counts: &Counts,
    discard_indexes: &[usize],
    improving_max_shanten: Option<i64>,
) -> Result<DiscardEvaluation, String> {
    evaluate_with(
        counts,
        discard_indexes,
        improving_max_shanten,
        |hand, total| core.numeric_shanten(hand, total),
    )
}

/// The enumeration itself, with the numeric shanten function injected so the
/// order, threshold and skipping rules are testable without the artifact.
fn evaluate_with<F>(
    counts: &Counts,
    discard_indexes: &[usize],
    improving_max_shanten: Option<i64>,
    mut shanten: F,
) -> Result<DiscardEvaluation, String>
where
    F: FnMut(&Counts, usize) -> Result<i32, String>,
{
    let total: usize = counts.iter().map(|&count| usize::from(count)).sum();
    debug_assert!(DISCARD_HAND_SIZES.contains(&total));
    debug_assert!(counts.iter().all(|&count| count <= MAX_COPIES));

    let mut hand = *counts;
    let mut evaluations = 0u64;
    let mut shanten_after = Vec::with_capacity(discard_indexes.len());
    let mut improving_after = Vec::with_capacity(discard_indexes.len());
    for &index in discard_indexes {
        debug_assert!(hand[index] > 0);
        hand[index] -= 1;
        evaluations += 1;
        let after = shanten(&hand, total - 1)?;
        let evaluate_improving =
            improving_max_shanten.is_none_or(|maximum| i64::from(after) <= maximum);
        let improving = if evaluate_improving {
            let mut found = Vec::new();
            for drawn in 0..TILE_KIND_COUNT {
                if hand[drawn] >= MAX_COPIES {
                    continue;
                }
                hand[drawn] += 1;
                evaluations += 1;
                let value = shanten(&hand, total);
                hand[drawn] -= 1;
                if value? < after {
                    found.push(drawn);
                }
            }
            Some(found)
        } else {
            None
        };
        hand[index] += 1;
        shanten_after.push(after);
        improving_after.push(improving);
    }
    Ok(DiscardEvaluation {
        shanten_after,
        improving_after,
        shanten_evaluations: evaluations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seven_pairs_shanten;

    /// Hand-computable stand-in for the numeric core: seven-pairs shanten.
    fn seven_pairs(hand: &Counts, total: usize) -> Result<i32, String> {
        assert_eq!(hand.iter().map(|&c| usize::from(c)).sum::<usize>(), total);
        assert!(hand.iter().all(|&count| count <= MAX_COPIES));
        Ok(seven_pairs_shanten(hand))
    }

    /// 11 22 33 44 55 66 7 8 (14 tiles): six pairs plus two singles.
    fn six_pairs_two_singles() -> Counts {
        let mut counts = [0; TILE_KIND_COUNT];
        counts[..6].fill(2);
        counts[6] = 1;
        counts[7] = 1;
        counts
    }

    #[test]
    fn results_follow_candidate_order_with_ascending_improving_tiles() {
        let counts = six_pairs_two_singles();
        // Discarding a single leaves 6 pairs / 7 kinds (shanten 0) that only
        // pairing the other single lowers.  Discarding a pair tile leaves
        // 5 pairs / 8 kinds (shanten 1) that pairing any single lowers.
        let result = evaluate_with(&counts, &[7, 0, 6], None, seven_pairs).unwrap();
        assert_eq!(
            result,
            DiscardEvaluation {
                shanten_after: vec![0, 1, 0],
                improving_after: vec![Some(vec![6]), Some(vec![0, 6, 7]), Some(vec![7])],
                shanten_evaluations: 3 + 34 * 3,
            }
        );
        assert_eq!(counts, six_pairs_two_singles());
    }

    #[test]
    fn threshold_is_inclusive_and_unevaluated_candidates_are_none() {
        let counts = six_pairs_two_singles();
        let at_zero = evaluate_with(&counts, &[0, 6], Some(0), seven_pairs).unwrap();
        assert_eq!(at_zero.shanten_after, vec![1, 0]);
        assert_eq!(at_zero.improving_after, vec![None, Some(vec![7])]);
        assert_eq!(at_zero.shanten_evaluations, 2 + 34);

        let below = evaluate_with(&counts, &[0, 6], Some(-1), seven_pairs).unwrap();
        assert_eq!(below.shanten_after, vec![1, 0]);
        assert_eq!(below.improving_after, vec![None, None]);
        assert_eq!(below.shanten_evaluations, 2);

        let above = evaluate_with(&counts, &[0, 6], Some(1), seven_pairs).unwrap();
        assert_eq!(
            above.improving_after,
            vec![Some(vec![0, 6, 7]), Some(vec![7])]
        );
        assert_eq!(counts, six_pairs_two_singles());
    }

    #[test]
    fn an_evaluated_empty_set_is_distinct_from_none() {
        let counts = six_pairs_two_singles();
        let constant = |_: &Counts, _: usize| Ok(3);
        let result = evaluate_with(&counts, &[0], None, constant).unwrap();
        assert_eq!(result.improving_after, vec![Some(vec![])]);
        let result = evaluate_with(&counts, &[0], Some(2), constant).unwrap();
        assert_eq!(result.improving_after, vec![None]);
    }

    #[test]
    fn no_candidates_evaluate_nothing() {
        let counts = six_pairs_two_singles();
        let result = evaluate_with(&counts, &[], None, |_, _| unreachable!()).unwrap();
        assert_eq!(
            result,
            DiscardEvaluation {
                shanten_after: vec![],
                improving_after: vec![],
                shanten_evaluations: 0,
            }
        );
    }

    #[test]
    fn types_held_four_times_after_the_discard_are_not_drawn() {
        // 1111 2 (5 tiles).  After discarding 2 the hand holds four 1s, so 1 is
        // skipped (33 draws); after discarding a 1 every type is drawn.
        let mut counts = [0; TILE_KIND_COUNT];
        counts[0] = 4;
        counts[1] = 1;
        let mut drawn_hands = Vec::new();
        let result = evaluate_with(&counts, &[1, 0], None, |hand, total| {
            drawn_hands.push((*hand, total));
            seven_pairs(hand, total)
        })
        .unwrap();
        // {1:4}: 1 pair / 1 kind = 11; any new kind gives 10.
        // {1:3, 2:1}: 1 pair / 2 kinds = 10; pairing 2 or any new kind gives 9.
        let without_first: Vec<usize> = (1..TILE_KIND_COUNT).collect();
        assert_eq!(result.shanten_after, vec![11, 10]);
        assert_eq!(
            result.improving_after,
            vec![Some(without_first.clone()), Some(without_first)]
        );
        assert_eq!(result.shanten_evaluations, (1 + 33) + (1 + 34));
        assert_eq!(drawn_hands.len() as u64, result.shanten_evaluations);
        assert!(drawn_hands[..34].iter().all(|(hand, _)| hand[0] == 4));
        assert_eq!(counts[0], 4);
        assert_eq!(counts[1], 1);
    }

    #[test]
    fn the_first_table_error_is_returned_without_a_partial_result() {
        let counts = six_pairs_two_singles();
        let mut calls = 0;
        let result = evaluate_with(&counts, &[0, 6], None, |hand, total| {
            calls += 1;
            if calls == 5 {
                Err("broken table".to_owned())
            } else {
                seven_pairs(hand, total)
            }
        });
        assert_eq!(result, Err("broken table".to_owned()));
        assert_eq!(calls, 5);
        assert_eq!(counts, six_pairs_two_singles());
    }
}
