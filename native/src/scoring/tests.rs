//! Hand-computed boundary cases (Issue #263).  The expected values below are
//! derived by hand from the standard rules, independently of lisjong-engine.

use super::*;

fn tiles(notation: &str) -> Vec<TileInput> {
    let mut result = Vec::new();
    let mut pending: Vec<u8> = Vec::new();
    for character in notation.chars() {
        if character == ' ' {
            continue;
        }
        if let Some(digit) = character.to_digit(10) {
            pending.push(digit as u8);
            continue;
        }
        let offset = match character {
            'm' => 0,
            'p' => 9,
            's' => 18,
            'z' => 27,
            _ => panic!("bad suit {character}"),
        };
        for digit in pending.drain(..) {
            let red = digit == 0;
            let rank = if red { 5 } else { digit };
            result.push(TileInput {
                kind: offset + rank - 1,
                red,
            });
        }
    }
    assert!(pending.is_empty(), "digits without suit in {notation}");
    result
}

fn tile(notation: &str) -> TileInput {
    let parsed = tiles(notation);
    assert_eq!(parsed.len(), 1);
    parsed[0]
}

fn meld(kind: MeldKind, notation: &str) -> MeldInput {
    MeldInput {
        kind,
        tiles: tiles(notation),
    }
}

/// Non-dealer (south) ron in the east round, no riichi, no dora.
fn ron(concealed: &str, melds: Vec<MeldInput>, winning: &str) -> WinInput {
    WinInput {
        concealed: tiles(concealed),
        melds,
        winning_tile: tile(winning),
        method: WinMethod::Ron,
        seat_wind: 1,
        prevailing_wind: 0,
        riichi: RiichiStatus::None,
        ippatsu: false,
        situation: WinSituation::Normal,
        dora_indicators: Vec::new(),
        ura_dora: UraDora::Indicators(Vec::new()),
    }
}

fn tsumo(concealed: &str, melds: Vec<MeldInput>, winning: &str) -> WinInput {
    WinInput {
        method: WinMethod::Tsumo,
        ..ron(concealed, melds, winning)
    }
}

fn riichi(mut input: WinInput, ura: &str) -> WinInput {
    input.riichi = RiichiStatus::Riichi;
    input.ura_dora = UraDora::Indicators(tiles(ura));
    input
}

fn standard() -> Rules {
    Rules::project_standard()
}

fn scored(input: &WinInput, rules: &Rules) -> Score {
    match evaluate(input, rules).expect("valid input") {
        Evaluation::Scored(score) => *score,
        other => panic!("expected a scored win, got {other:?}"),
    }
}

fn yaku_names(score: &Score) -> Vec<&'static str> {
    score.yaku.iter().map(|value| value.yaku.name()).collect()
}

#[test]
fn seven_pairs_is_fixed_25_fu() {
    let score = scored(&ron("1133m5577p2266s7z", vec![], "7z"), &standard());
    assert_eq!(score.shape, WinningShape::SevenPairs);
    assert_eq!(yaku_names(&score), ["chiitoitsu"]);
    assert_eq!(score.fu, Some(25));
    assert_eq!(score.fu_components, [(FuReason::SevenPairs, 25)]);
    // 25 * 2^(2+2) = 400 -> 1600.
    assert_eq!(score.ron_payment, Some(1600));
}

#[test]
fn pinfu_tsumo_is_20_fu_without_tsumo_fu() {
    let score = scored(&tsumo("23m567p345s678s22p", vec![], "1m"), &standard());
    assert_eq!(yaku_names(&score), ["menzen_tsumo", "pinfu"]);
    assert_eq!(score.fu, Some(20));
    assert_eq!(score.fu_components, [(FuReason::Base, 20)]);
    // 20 * 2^4 = 320: dealer pays 640 -> 700, non-dealers 320 -> 400.
    assert_eq!(score.tsumo_dealer_payment, Some(700));
    assert_eq!(score.tsumo_non_dealer_payment, Some(400));
    assert_eq!(score.winner_points, 1500);
}

#[test]
fn menzen_ron_adds_10_fu() {
    let score = scored(
        &riichi(ron("23m567p345s678s22p", vec![], "1m"), ""),
        &standard(),
    );
    assert_eq!(yaku_names(&score), ["riichi", "pinfu"]);
    assert_eq!(
        score.fu_components,
        [(FuReason::Base, 20), (FuReason::MenzenRon, 10)]
    );
    assert_eq!(score.fu, Some(30));
    assert_eq!(score.ron_payment, Some(2000));
}

#[test]
fn ron_assignment_tanki_versus_ryanmen() {
    // 11m + 123m: the 1m completes either the pair (tanki) or 123m (ryanmen).
    let input = ron("1123m456p789p234s", vec![], "1m");
    let score = scored(&input, &standard());
    assert_eq!(score.wait, WaitType::Ryanmen);
    assert_eq!(yaku_names(&score), ["pinfu"]);
    assert_eq!(score.ron_payment, Some(1000));

    // With riichi: pinfu + riichi 2 han 30 fu (2000) beats tanki riichi
    // 1 han 40 fu (1300).
    let score = scored(&riichi(input, ""), &standard());
    assert_eq!(score.wait, WaitType::Ryanmen);
    assert_eq!(score.ron_payment, Some(2000));
    assert_eq!(score.completed_group, Some(0));
}

#[test]
fn ron_completed_triplet_is_open_and_breaks_suuankou() {
    let input = ron("222m333p444s55s66s", vec![], "5s");
    let score = scored(&input, &standard());
    assert_eq!(yaku_names(&score), ["tanyao", "toitoi", "sanankou"]);
    assert_eq!(
        score.fu_components,
        [
            (FuReason::Base, 20),
            (FuReason::MenzenRon, 10),
            (FuReason::ClosedSimpleTriplet, 4),
            (FuReason::ClosedSimpleTriplet, 4),
            (FuReason::ClosedSimpleTriplet, 4),
            (FuReason::OpenSimpleTriplet, 2),
        ]
    );
    assert_eq!(score.fu, Some(50));
    assert_eq!(score.limit, ScoreLimit::Mangan);
    assert_eq!(score.ron_payment, Some(8000));

    let score = scored(&tsumo("222m333p444s55s66s", vec![], "5s"), &standard());
    assert_eq!(yaku_names(&score), ["suuankou"]);
    assert_eq!(score.dora, None);
    assert_eq!(score.winner_points, 32000);
}

#[test]
fn dora_alone_is_not_a_yaku() {
    let mut input = ron("123p456p789s2s", vec![meld(MeldKind::Pon, "999m")], "2s");
    input.dora_indicators = tiles("1s");
    assert_eq!(
        evaluate(&input, &standard()),
        Ok(Evaluation::CompleteWithoutYaku)
    );
}

#[test]
fn ryanpeikou_beats_seven_pairs() {
    let score = scored(&ron("112233m445566p7s", vec![], "7s"), &standard());
    assert_eq!(score.shape, WinningShape::Standard);
    assert_eq!(yaku_names(&score), ["ryanpeikou"]);
    // 20 + 10 + tanki 2 = 32 -> 40 fu; 40 * 2^5 = 1280 -> 5200.
    assert_eq!(score.fu, Some(40));
    assert_eq!(score.ron_payment, Some(5200));
}

#[test]
fn compound_yakuman_and_rule_switches() {
    let input = ron("555z666z777z111z2z", vec![], "2z");
    let score = scored(&input, &standard());
    assert_eq!(
        yaku_names(&score),
        ["daisangen", "suuankou_tanki", "tsuuiisou"]
    );
    assert_eq!(score.yakuman_units, 3);
    assert_eq!(score.ron_payment, Some(96000));

    let single = Rules {
        multiple_yakuman_enabled: false,
        ..standard()
    };
    assert_eq!(scored(&input, &single).ron_payment, Some(32000));

    let double = Rules {
        double_yakuman_variants: vec![Yaku::SuuankouTanki],
        ..standard()
    };
    let score = scored(&input, &double);
    assert_eq!(score.yakuman_units, 4);
    assert_eq!(score.ron_payment, Some(128000));
}

#[test]
fn counted_yakuman_switch() {
    // riichi 1 + tsumo 1 + pinfu 1 + iipeikou 1 + ittsuu 2 + chinitsu 6 = 12,
    // dora 9m x3 = 15 han.
    let mut input = riichi(tsumo("1122334567899m", vec![], "9m"), "1z");
    input.dora_indicators = tiles("8m");
    let score = scored(&input, &standard());
    assert_eq!(score.han, 15);
    assert_eq!(score.limit, ScoreLimit::Yakuman);
    assert_eq!(score.yakuman_units, 0);
    assert!(score.fu.is_some());
    assert_eq!(score.winner_points, 32000);

    let rules = Rules {
        counted_yakuman_enabled: false,
        ..standard()
    };
    let score = scored(&input, &rules);
    assert_eq!(score.limit, ScoreLimit::Sanbaiman);
    assert_eq!(score.winner_points, 24000);
}

#[test]
fn rounded_mangan_and_ura_dora_modes() {
    // riichi + pinfu + tanyao + dora 2m = 4 han 30 fu.
    let mut input = riichi(ron("234m567p345s66s78s", vec![], "6s"), "1z");
    input.dora_indicators = tiles("1m");
    let score = scored(&input, &standard());
    assert_eq!(score.han, 4);
    assert_eq!(score.fu, Some(30));
    assert_eq!(
        score.dora,
        Some(DoraCount {
            dora: 1,
            red: 0,
            ura: Some(0)
        })
    );
    assert_eq!(score.ron_payment, Some(7700));
    let rules = Rules {
        rounded_mangan_enabled: true,
        ..standard()
    };
    assert_eq!(scored(&input, &rules).ron_payment, Some(8000));

    // A concrete ura indicator 5s makes 6s dora: 3 more han.
    input.ura_dora = UraDora::Indicators(tiles("5s"));
    let score = scored(&input, &standard());
    assert_eq!(score.dora.unwrap().ura, Some(3));
    assert_eq!(score.ron_payment, Some(12000));
    assert!(!score.ura_dora_excluded);

    // Excluded mode never counts ura dora and says so.
    input.ura_dora = UraDora::Excluded;
    let score = scored(&input, &standard());
    assert_eq!(score.dora.unwrap().ura, None);
    assert!(score.ura_dora_excluded);
    assert_eq!(score.ron_payment, Some(7700));
}

#[test]
fn kuisagari_reduces_open_han() {
    let open = ron("456m789m234p5s", vec![meld(MeldKind::Chi, "123m")], "5s");
    let score = scored(&open, &standard());
    assert_eq!(yaku_names(&score), ["ittsuu"]);
    assert_eq!(score.yaku_han, 1);
    // 20 + tanki 2 = 22 -> 30 fu.
    assert_eq!(score.fu, Some(30));
    assert_eq!(score.ron_payment, Some(1000));

    let closed = ron("123456789m234p5s", vec![], "5s");
    let score = scored(&closed, &standard());
    assert_eq!(score.yaku_han, 2);
    assert_eq!(score.fu, Some(40));
    assert_eq!(score.ron_payment, Some(2600));
}

#[test]
fn kuitan_switch() {
    let input = ron("234m567m345s6s", vec![meld(MeldKind::Pon, "222p")], "6s");
    let score = scored(&input, &standard());
    assert_eq!(yaku_names(&score), ["tanyao"]);
    assert_eq!(score.fu, Some(30));
    assert_eq!(score.ron_payment, Some(1000));

    let rules = Rules {
        kuitan_enabled: false,
        ..standard()
    };
    assert_eq!(
        evaluate(&input, &rules),
        Ok(Evaluation::CompleteWithoutYaku)
    );
}

#[test]
fn red_fives_count_and_disabled_red_rejects_them() {
    let input = ron("234m0p67p345s678s2s", vec![], "2s");
    let score = scored(&input, &standard());
    // tanyao only (tanki wait), red 1: 2 han 40 fu = 2600.
    assert_eq!(score.dora.unwrap().red, 1);
    assert_eq!(score.han, 2);
    assert_eq!(score.ron_payment, Some(2600));
    let rules = Rules {
        red_dora_enabled: false,
        ..standard()
    };
    assert!(evaluate(&input, &rules).is_err());
}

#[test]
fn plain_fives_are_limited_to_three_per_suit_with_red_dora() {
    // 5555m as ankan: four plain fives cannot exist when red dora is enabled.
    let four_plain = tsumo("234p567p345s6s", vec![meld(MeldKind::Ankan, "5555m")], "6s");
    assert!(evaluate(&four_plain, &standard()).is_err());

    // The same count with an indicator tile also fails (indicators are counted).
    let mut with_indicator = tsumo("555m234p567p345s6s", vec![], "6s");
    with_indicator.dora_indicators = tiles("5m");
    assert!(evaluate(&with_indicator, &standard()).is_err());

    // Three plain fives plus the red five are allowed.
    let with_red = tsumo("234p567p345s6s", vec![meld(MeldKind::Ankan, "0555m")], "6s");
    assert!(evaluate(&with_red, &standard()).is_ok());

    // Without red dora all four fives are plain.
    let rules = Rules {
        red_dora_enabled: false,
        ..standard()
    };
    assert!(evaluate(&four_plain, &rules).is_ok());
}

#[test]
fn dealer_payments() {
    let mut input = ron("23m567p345s678s22p", vec![], "1m");
    input.seat_wind = 0;
    // pinfu 1 han 30 fu: 240 * 6 = 1440 -> 1500.
    assert_eq!(scored(&input, &standard()).ron_payment, Some(1500));
    input.method = WinMethod::Tsumo;
    let score = scored(&input, &standard());
    // tsumo + pinfu 2 han 20 fu: 320 * 2 = 640 -> 700 all.
    assert_eq!(score.tsumo_dealer_payment, None);
    assert_eq!(score.tsumo_non_dealer_payment, Some(700));
    assert_eq!(score.winner_points, 2100);
}

#[test]
fn thirteen_sided_kokushi() {
    let score = scored(&ron("19m19p19s1234567z", vec![], "1m"), &standard());
    assert_eq!(score.shape, WinningShape::ThirteenOrphans);
    assert_eq!(score.wait, WaitType::KokushiThirteenSided);
    assert_eq!(yaku_names(&score), ["kokushi_musou_13_wait"]);
    assert_eq!(score.ron_payment, Some(32000));
}

#[test]
fn not_complete_hand() {
    let input = ron("1235m567p345s678s", vec![], "9m");
    assert_eq!(evaluate(&input, &standard()), Ok(Evaluation::NotComplete));
}

#[test]
fn invalid_inputs_fail_closed() {
    let valid = ron("23m567p345s678s22p", vec![], "1m");
    assert!(evaluate(&valid, &standard()).is_ok());

    let short = ron("23m567p345s678s2p", vec![], "1m");
    assert!(evaluate(&short, &standard()).is_err());

    let five_copies = ron("1111m567p345s678s2p", vec![], "1m");
    assert!(evaluate(&five_copies, &standard()).is_err());

    let mut open_riichi = ron("456m789m234p5s", vec![meld(MeldKind::Chi, "123m")], "5s");
    open_riichi.riichi = RiichiStatus::Riichi;
    assert!(evaluate(&open_riichi, &standard()).is_err());

    let mut ippatsu = valid.clone();
    ippatsu.ippatsu = true;
    assert!(evaluate(&ippatsu, &standard()).is_err());

    let mut bad_ura = riichi(valid.clone(), "");
    bad_ura.dora_indicators = tiles("1z");
    assert!(evaluate(&bad_ura, &standard()).is_err());

    let mut ura_without_riichi = valid.clone();
    ura_without_riichi.ura_dora = UraDora::Indicators(tiles("1z"));
    assert!(evaluate(&ura_without_riichi, &standard()).is_err());

    let mut haitei_ron = valid.clone();
    haitei_ron.situation = WinSituation::Haitei;
    assert!(evaluate(&haitei_ron, &standard()).is_err());

    let mut rinshan_without_kan = tsumo("23m567p345s678s22p", vec![], "1m");
    rinshan_without_kan.situation = WinSituation::Rinshan;
    assert!(evaluate(&rinshan_without_kan, &standard()).is_err());

    let mut dealer_chiihou = tsumo("23m567p345s678s22p", vec![], "1m");
    dealer_chiihou.seat_wind = 0;
    dealer_chiihou.situation = WinSituation::Chiihou;
    assert!(evaluate(&dealer_chiihou, &standard()).is_err());

    let bad_chi = ron("456m789m234p5s", vec![meld(MeldKind::Chi, "124m")], "5s");
    assert!(evaluate(&bad_chi, &standard()).is_err());

    let two_red = ron("234m0p67p345s678s0p", vec![], "2s");
    assert!(evaluate(&two_red, &standard()).is_err());
}

#[test]
fn situational_yaku_and_tenhou() {
    let mut input = tsumo("23m567p345s678s22p", vec![], "1m");
    input.situation = WinSituation::Chiihou;
    let score = scored(&input, &standard());
    assert_eq!(yaku_names(&score), ["chiihou"]);

    let mut haitei = tsumo("456m789m234p5s", vec![meld(MeldKind::Chi, "123m")], "5s");
    haitei.situation = WinSituation::Haitei;
    assert_eq!(
        yaku_names(&scored(&haitei, &standard())),
        ["haitei", "ittsuu"]
    );
}

#[test]
fn shapes_choose_identically_regardless_of_tile_order() {
    let forward = ron("1123m456p789p234s", vec![], "1m");
    let mut reversed = forward.clone();
    reversed.concealed.reverse();
    assert_eq!(
        evaluate(&forward, &standard()),
        evaluate(&reversed, &standard())
    );
}
