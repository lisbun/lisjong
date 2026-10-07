//! AI-side yaku / fu / score calculator (Issue #263).
//!
//! A pure calculator for one concrete complete hand plus an explicit winning
//! context.  It never estimates anything: unknown values (for example ura dora
//! that is not known online) must be passed explicitly, either as concrete
//! indicators or as an explicit exclusion.
//!
//! Results distinguish invalid input (`Err`), a non-complete hand, a complete
//! hand without yaku (dora alone never makes a yaku) and a scored win.  Among
//! every interpretation (shape x decomposition x winning-tile assignment) the
//! one with the highest winner payment is chosen; ties are broken by the fixed
//! order documented on [`evaluate`].
//!
//! Honba, riichi sticks and pao (responsibility payment) are not part of the
//! result: round settlement belongs to `lisjong-engine`.  The semantics follow
//! `lisjong-engine`'s scoring layer (verified with fixed engine fixtures), but
//! there is no runtime dependency on the engine.
//!
//! Tile kinds use the common 34-kind ids: 0..=8 manzu, 9..=17 pinzu,
//! 18..=26 souzu, 27..=30 east/south/west/north, 31..=33 white/green/red.

use std::cmp::Ordering;

pub const TILE_KIND_COUNT: usize = 34;
const COPIES_PER_KIND: u8 = 4;
const MAX_MELDS: usize = 4;
const MAX_DORA_INDICATORS: usize = 5;
const RED_FIVE_KINDS: [u8; 3] = [4, 13, 22];
const FIRST_HONOR: u8 = 27;
const FIRST_DRAGON: u8 = 31;
const GREEN_KINDS: [u8; 6] = [19, 20, 21, 23, 25, 32];
const CHUUREN_BASE_COUNTS: [u8; 9] = [3, 1, 1, 1, 1, 1, 1, 1, 3];
const KOKUSHI_KINDS: [u8; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileInput {
    pub kind: u8,
    pub red: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeldKind {
    Chi,
    Pon,
    Daiminkan,
    Ankan,
    Kakan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeldInput {
    pub kind: MeldKind,
    pub tiles: Vec<TileInput>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WinMethod {
    Ron,
    Tsumo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiichiStatus {
    None,
    Riichi,
    DoubleRiichi,
}

/// Situational win timing.  These are mutually exclusive by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WinSituation {
    Normal,
    Haitei,
    Houtei,
    Rinshan,
    Chankan,
    Tenhou,
    Chiihou,
}

/// Ura dora mode.  `Excluded` scores without ura dora and marks the result;
/// it is never treated as "zero ura dora".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UraDora {
    Indicators(Vec<TileInput>),
    Excluded,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WinInput {
    /// Concealed tiles, excluding the winning tile.
    pub concealed: Vec<TileInput>,
    pub melds: Vec<MeldInput>,
    pub winning_tile: TileInput,
    pub method: WinMethod,
    /// 0 = east .. 3 = north.  East is the dealer.
    pub seat_wind: u8,
    pub prevailing_wind: u8,
    pub riichi: RiichiStatus,
    pub ippatsu: bool,
    pub situation: WinSituation,
    /// Dora indicators (the normal indicator followed by kan indicators).
    pub dora_indicators: Vec<TileInput>,
    pub ura_dora: UraDora,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rules {
    pub kuitan_enabled: bool,
    pub red_dora_enabled: bool,
    pub rounded_mangan_enabled: bool,
    pub counted_yakuman_enabled: bool,
    pub multiple_yakuman_enabled: bool,
    pub double_yakuman_variants: Vec<Yaku>,
    pub double_wind_pair_fu: u8,
}

impl Rules {
    /// The same conditions as `lisjong-engine` `PROJECT_STANDARD_RULES`.
    pub fn project_standard() -> Self {
        Rules {
            kuitan_enabled: true,
            red_dora_enabled: true,
            rounded_mangan_enabled: false,
            counted_yakuman_enabled: true,
            multiple_yakuman_enabled: true,
            double_yakuman_variants: Vec::new(),
            double_wind_pair_fu: 4,
        }
    }
}

macro_rules! yaku_table {
    ($($variant:ident => $name:literal, $closed:expr, $open:expr, $yakuman:expr;)*) => {
        /// Yaku identifiers, in the fixed output order (same as lisjong-engine).
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum Yaku { $($variant),* }

        impl Yaku {
            pub const ALL: &'static [Yaku] = &[$(Yaku::$variant),*];

            pub fn name(self) -> &'static str {
                match self { $(Yaku::$variant => $name),* }
            }

            fn han(self, menzen: bool) -> Option<u8> {
                match self { $(Yaku::$variant => if menzen { $closed } else { $open }),* }
            }

            pub fn is_yakuman(self) -> bool {
                match self { $(Yaku::$variant => $yakuman),* }
            }
        }
    };
}

yaku_table! {
    MenzenTsumo => "menzen_tsumo", Some(1), None, false;
    Riichi => "riichi", Some(1), None, false;
    Ippatsu => "ippatsu", Some(1), None, false;
    DoubleRiichi => "double_riichi", Some(2), None, false;
    Chankan => "chankan", Some(1), Some(1), false;
    RinshanKaihou => "rinshan_kaihou", Some(1), Some(1), false;
    Haitei => "haitei", Some(1), Some(1), false;
    Houtei => "houtei", Some(1), Some(1), false;
    Tanyao => "tanyao", Some(1), Some(1), false;
    SeatWind => "seat_wind", Some(1), Some(1), false;
    PrevailingWind => "prevailing_wind", Some(1), Some(1), false;
    WhiteDragon => "white_dragon", Some(1), Some(1), false;
    GreenDragon => "green_dragon", Some(1), Some(1), false;
    RedDragon => "red_dragon", Some(1), Some(1), false;
    Pinfu => "pinfu", Some(1), None, false;
    Iipeikou => "iipeikou", Some(1), None, false;
    Chanta => "chanta", Some(2), Some(1), false;
    Ittsuu => "ittsuu", Some(2), Some(1), false;
    SanshokuDoujun => "sanshoku_doujun", Some(2), Some(1), false;
    SanshokuDoukou => "sanshoku_doukou", Some(2), Some(2), false;
    Sankantsu => "sankantsu", Some(2), Some(2), false;
    Toitoi => "toitoi", Some(2), Some(2), false;
    Sanankou => "sanankou", Some(2), Some(2), false;
    Shousangen => "shousangen", Some(2), Some(2), false;
    Honroutou => "honroutou", Some(2), Some(2), false;
    Chiitoitsu => "chiitoitsu", Some(2), None, false;
    Junchan => "junchan", Some(3), Some(2), false;
    Honitsu => "honitsu", Some(3), Some(2), false;
    Ryanpeikou => "ryanpeikou", Some(3), None, false;
    Chinitsu => "chinitsu", Some(6), Some(5), false;
    Tenhou => "tenhou", None, None, true;
    Chiihou => "chiihou", None, None, true;
    Daisangen => "daisangen", None, None, true;
    Suuankou => "suuankou", None, None, true;
    SuuankouTanki => "suuankou_tanki", None, None, true;
    Tsuuiisou => "tsuuiisou", None, None, true;
    Ryuuiisou => "ryuuiisou", None, None, true;
    Chinroutou => "chinroutou", None, None, true;
    KokushiMusou => "kokushi_musou", None, None, true;
    KokushiMusou13Wait => "kokushi_musou_13_wait", None, None, true;
    Daisuushii => "daisuushii", None, None, true;
    Shousuushii => "shousuushii", None, None, true;
    Suukantsu => "suukantsu", None, None, true;
    ChuurenPoutou => "chuuren_poutou", None, None, true;
    JunseiChuurenPoutou => "junsei_chuuren_poutou", None, None, true;
}

/// Yaku that a rule may count as double yakuman (same set as lisjong-engine).
pub const DOUBLE_YAKUMAN_CANDIDATES: [Yaku; 4] = [
    Yaku::SuuankouTanki,
    Yaku::KokushiMusou13Wait,
    Yaku::Daisuushii,
    Yaku::JunseiChuurenPoutou,
];

impl Yaku {
    pub fn from_name(name: &str) -> Option<Yaku> {
        Yaku::ALL.iter().copied().find(|yaku| yaku.name() == name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum WinningShape {
    Standard,
    SevenPairs,
    ThirteenOrphans,
}

impl WinningShape {
    pub fn name(self) -> &'static str {
        match self {
            WinningShape::Standard => "standard",
            WinningShape::SevenPairs => "seven_pairs",
            WinningShape::ThirteenOrphans => "thirteen_orphans",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitType {
    Ryanmen,
    Kanchan,
    Penchan,
    Shanpon,
    Tanki,
    KokushiSingle,
    KokushiThirteenSided,
}

impl WaitType {
    pub fn name(self) -> &'static str {
        match self {
            WaitType::Ryanmen => "ryanmen",
            WaitType::Kanchan => "kanchan",
            WaitType::Penchan => "penchan",
            WaitType::Shanpon => "shanpon",
            WaitType::Tanki => "tanki",
            WaitType::KokushiSingle => "kokushi_single",
            WaitType::KokushiThirteenSided => "kokushi_thirteen_sided",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GroupKind {
    Triplet,
    Sequence,
    Quad,
}

impl GroupKind {
    pub fn name(self) -> &'static str {
        match self {
            GroupKind::Sequence => "sequence",
            GroupKind::Triplet => "triplet",
            GroupKind::Quad => "quad",
        }
    }
}

/// One group of a standard interpretation.  `kind_id` is the sequence start
/// tile for sequences.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Group {
    pub kind: GroupKind,
    pub tile: u8,
    /// A declared meld other than ankan.
    pub is_open: bool,
    /// A concealed triplet completed by the ron tile (scored as open).
    pub completed_by_ron: bool,
}

impl Group {
    pub fn is_concealed_for_scoring(&self) -> bool {
        !self.is_open && !self.completed_by_ron
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FuReason {
    SevenPairs,
    Base,
    MenzenRon,
    Tsumo,
    DragonPair,
    SeatWindPair,
    PrevailingWindPair,
    DoubleWindPair,
    OpenSimpleTriplet,
    OpenTerminalOrHonorTriplet,
    ClosedSimpleTriplet,
    ClosedTerminalOrHonorTriplet,
    OpenSimpleQuad,
    OpenTerminalOrHonorQuad,
    ClosedSimpleQuad,
    ClosedTerminalOrHonorQuad,
    KanchanWait,
    PenchanWait,
    TankiWait,
}

impl FuReason {
    pub fn name(self) -> &'static str {
        match self {
            FuReason::SevenPairs => "seven_pairs",
            FuReason::Base => "base",
            FuReason::MenzenRon => "menzen_ron",
            FuReason::Tsumo => "tsumo",
            FuReason::DragonPair => "dragon_pair",
            FuReason::SeatWindPair => "seat_wind_pair",
            FuReason::PrevailingWindPair => "prevailing_wind_pair",
            FuReason::DoubleWindPair => "double_wind_pair",
            FuReason::OpenSimpleTriplet => "open_simple_triplet",
            FuReason::OpenTerminalOrHonorTriplet => "open_terminal_or_honor_triplet",
            FuReason::ClosedSimpleTriplet => "closed_simple_triplet",
            FuReason::ClosedTerminalOrHonorTriplet => "closed_terminal_or_honor_triplet",
            FuReason::OpenSimpleQuad => "open_simple_quad",
            FuReason::OpenTerminalOrHonorQuad => "open_terminal_or_honor_quad",
            FuReason::ClosedSimpleQuad => "closed_simple_quad",
            FuReason::ClosedTerminalOrHonorQuad => "closed_terminal_or_honor_quad",
            FuReason::KanchanWait => "kanchan_wait",
            FuReason::PenchanWait => "penchan_wait",
            FuReason::TankiWait => "tanki_wait",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreLimit {
    None,
    Mangan,
    Haneman,
    Baiman,
    Sanbaiman,
    Yakuman,
}

impl ScoreLimit {
    pub fn name(self) -> &'static str {
        match self {
            ScoreLimit::None => "none",
            ScoreLimit::Mangan => "mangan",
            ScoreLimit::Haneman => "haneman",
            ScoreLimit::Baiman => "baiman",
            ScoreLimit::Sanbaiman => "sanbaiman",
            ScoreLimit::Yakuman => "yakuman",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YakuValue {
    pub yaku: Yaku,
    pub han: u8,
    pub yakuman_units: u8,
}

/// Dora bonus.  `ura` is `None` when ura dora was explicitly excluded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoraCount {
    pub dora: u8,
    pub red: u8,
    pub ura: Option<u8>,
}

impl DoraCount {
    pub fn total(&self) -> u32 {
        u32::from(self.dora) + u32::from(self.red) + u32::from(self.ura.unwrap_or(0))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Score {
    pub shape: WinningShape,
    pub wait: WaitType,
    /// Pair of a standard interpretation; `None` for the special shapes.
    pub pair: Option<u8>,
    /// Concealed groups (canonical order) followed by declared melds (input
    /// order).  Empty for the special shapes.
    pub groups: Vec<Group>,
    /// Index into `groups` of the group completed by the winning tile; `None`
    /// when the winning tile completes the pair or the shape is special.
    pub completed_group: Option<usize>,
    pub yaku: Vec<YakuValue>,
    /// Dora breakdown; `None` for yakuman (dora is never added to yakuman).
    pub dora: Option<DoraCount>,
    pub ura_dora_excluded: bool,
    pub yaku_han: u32,
    pub han: u32,
    pub fu: Option<u32>,
    pub fu_components: Vec<(FuReason, u32)>,
    pub yakuman_units: u32,
    pub is_dealer: bool,
    pub method: WinMethod,
    pub base_points: u32,
    pub limit: ScoreLimit,
    pub ron_payment: Option<u32>,
    pub tsumo_dealer_payment: Option<u32>,
    pub tsumo_non_dealer_payment: Option<u32>,
    pub winner_points: u32,
    /// Tie-break key among equal winner points (see `evaluate`).
    tie_key: TieKey,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TieKey {
    shape: WinningShape,
    pair: Option<u8>,
    concealed: Vec<(u8, GroupKind)>,
    completed: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Evaluation {
    NotComplete,
    CompleteWithoutYaku,
    Scored(Box<Score>),
}

fn is_suited(kind: u8) -> bool {
    kind < FIRST_HONOR
}

fn rank(kind: u8) -> u8 {
    kind % 9 + 1
}

fn is_terminal(kind: u8) -> bool {
    is_suited(kind) && matches!(rank(kind), 1 | 9)
}

fn is_terminal_or_honor(kind: u8) -> bool {
    !is_suited(kind) || is_terminal(kind)
}

fn is_simple(kind: u8) -> bool {
    is_suited(kind) && !is_terminal(kind)
}

fn suit(kind: u8) -> u8 {
    kind / 9
}

fn wind_kind(wind: u8) -> u8 {
    FIRST_HONOR + wind
}

fn dora_kind(indicator: u8) -> u8 {
    if is_suited(indicator) {
        suit(indicator) * 9 + rank(indicator) % 9
    } else if indicator < FIRST_DRAGON {
        FIRST_HONOR + (indicator - FIRST_HONOR + 1) % 4
    } else {
        FIRST_DRAGON + (indicator - FIRST_DRAGON + 1) % 3
    }
}

fn validate_tile(tile: &TileInput, rules: &Rules) -> Result<(), String> {
    if usize::from(tile.kind) >= TILE_KIND_COUNT {
        return Err("tile kind must be between 0 and 33".to_string());
    }
    if tile.red {
        if !RED_FIVE_KINDS.contains(&tile.kind) {
            return Err("only suited fives can be red".to_string());
        }
        if !rules.red_dora_enabled {
            return Err("red fives do not exist when red dora is disabled".to_string());
        }
    }
    Ok(())
}

fn validate_meld(meld: &MeldInput) -> Result<(), String> {
    let expected = match meld.kind {
        MeldKind::Chi | MeldKind::Pon => 3,
        MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => 4,
    };
    if meld.tiles.len() != expected {
        return Err(format!(
            "{:?} meld must contain {expected} tiles",
            meld.kind
        ));
    }
    let first = meld.tiles[0].kind;
    if meld.kind == MeldKind::Chi {
        if !is_suited(first) || meld.tiles.iter().any(|tile| suit(tile.kind) != suit(first)) {
            return Err("chi tiles must be suited tiles of one suit".to_string());
        }
        if meld.tiles.iter().any(|tile| !is_suited(tile.kind)) {
            return Err("chi tiles must be suited tiles of one suit".to_string());
        }
        let mut ranks: Vec<u8> = meld.tiles.iter().map(|tile| rank(tile.kind)).collect();
        ranks.sort_unstable();
        if ranks[1] != ranks[0] + 1 || ranks[2] != ranks[0] + 2 {
            return Err("chi tiles must have three consecutive ranks".to_string());
        }
    } else if meld.tiles.iter().any(|tile| tile.kind != first) {
        return Err(format!("{:?} tiles must share one tile kind", meld.kind));
    }
    Ok(())
}

fn meld_group(meld: &MeldInput) -> Group {
    let tile = meld.tiles.iter().map(|tile| tile.kind).min().unwrap_or(0);
    let kind = match meld.kind {
        MeldKind::Chi => GroupKind::Sequence,
        MeldKind::Pon => GroupKind::Triplet,
        MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => GroupKind::Quad,
    };
    Group {
        kind,
        tile,
        is_open: meld.kind != MeldKind::Ankan,
        completed_by_ron: false,
    }
}

fn validate(input: &WinInput, rules: &Rules) -> Result<(), String> {
    if !matches!(rules.double_wind_pair_fu, 2 | 4) {
        return Err("double_wind_pair_fu must be 2 or 4".to_string());
    }
    if rules
        .double_yakuman_variants
        .iter()
        .any(|yaku| !DOUBLE_YAKUMAN_CANDIDATES.contains(yaku))
    {
        return Err("double_yakuman_variants contains an unsupported yaku".to_string());
    }
    if input.melds.len() > MAX_MELDS {
        return Err("a hand can have at most four melds".to_string());
    }
    if input.concealed.len() + 1 + 3 * input.melds.len() != 14 {
        return Err(
            "concealed tiles must contain 13 - 3 * meld count tiles (winning tile excluded)"
                .to_string(),
        );
    }
    if input.seat_wind > 3 || input.prevailing_wind > 3 {
        return Err("winds must be between 0 and 3".to_string());
    }
    if input.dora_indicators.len() > MAX_DORA_INDICATORS {
        return Err("at most five dora indicators exist".to_string());
    }
    for meld in &input.melds {
        validate_meld(meld)?;
    }

    let ura_tiles: &[TileInput] = match &input.ura_dora {
        UraDora::Indicators(tiles) => tiles,
        UraDora::Excluded => &[],
    };
    let mut kind_counts = [0u8; TILE_KIND_COUNT];
    let mut red_counts = [0u8; 3];
    let mut plain_five_counts = [0u8; 3];
    let all_physical = input
        .concealed
        .iter()
        .chain(std::iter::once(&input.winning_tile))
        .chain(input.melds.iter().flat_map(|meld| meld.tiles.iter()))
        .chain(input.dora_indicators.iter())
        .chain(ura_tiles.iter());
    for tile in all_physical {
        validate_tile(tile, rules)?;
        kind_counts[usize::from(tile.kind)] += 1;
        if kind_counts[usize::from(tile.kind)] > COPIES_PER_KIND {
            return Err("no tile kind can appear more than four times".to_string());
        }
        if tile.red {
            let index = usize::from(suit(tile.kind));
            red_counts[index] += 1;
            if red_counts[index] > 1 {
                return Err("each suit has only one red five".to_string());
            }
        } else if rules.red_dora_enabled && RED_FIVE_KINDS.contains(&tile.kind) {
            // With red dora each suit's fives are one red and three plain
            // copies, so a fourth plain five is an input error, not a hand
            // that silently lacks its red dora.
            let index = usize::from(suit(tile.kind));
            plain_five_counts[index] += 1;
            if plain_five_counts[index] > COPIES_PER_KIND - 1 {
                return Err("with red dora each suit has only three plain fives".to_string());
            }
        }
    }

    let menzen = input.melds.iter().all(|meld| meld.kind == MeldKind::Ankan);
    if input.riichi != RiichiStatus::None && !menzen {
        return Err("riichi requires a menzen hand".to_string());
    }
    if input.ippatsu && input.riichi == RiichiStatus::None {
        return Err("ippatsu requires riichi".to_string());
    }
    match input.situation {
        WinSituation::Normal => {}
        WinSituation::Haitei => {
            if input.method != WinMethod::Tsumo {
                return Err("haitei requires tsumo".to_string());
            }
        }
        WinSituation::Houtei => {
            if input.method != WinMethod::Ron {
                return Err("houtei requires ron".to_string());
            }
        }
        WinSituation::Rinshan => {
            if input.method != WinMethod::Tsumo {
                return Err("rinshan requires tsumo".to_string());
            }
            if !input
                .melds
                .iter()
                .any(|meld| meld_group(meld).kind == GroupKind::Quad)
            {
                return Err("rinshan requires a declared kan".to_string());
            }
            if input.ippatsu {
                return Err("ippatsu and rinshan cannot coexist".to_string());
            }
        }
        WinSituation::Chankan => {
            if input.method != WinMethod::Ron {
                return Err("chankan requires ron".to_string());
            }
        }
        WinSituation::Tenhou | WinSituation::Chiihou => {
            if input.method != WinMethod::Tsumo || !input.melds.is_empty() {
                return Err("tenhou / chiihou require an uncalled tsumo".to_string());
            }
            if input.riichi != RiichiStatus::None || input.ippatsu {
                return Err("tenhou / chiihou cannot have riichi or ippatsu".to_string());
            }
            let is_dealer = input.seat_wind == 0;
            if (input.situation == WinSituation::Tenhou) != is_dealer {
                return Err("tenhou is the dealer's and chiihou a non-dealer's".to_string());
            }
        }
    }
    if let UraDora::Indicators(tiles) = &input.ura_dora {
        if input.riichi == RiichiStatus::None {
            if !tiles.is_empty() {
                return Err("ura dora indicators require riichi".to_string());
            }
        } else if tiles.len() != input.dora_indicators.len() {
            return Err("ura dora indicators must correspond to dora indicators".to_string());
        }
    }
    Ok(())
}

/// Every standard decomposition of `counts` into one pair and `groups`
/// concealed groups, each in canonical order (lowest tile first, triplet tried
/// before sequence).
fn standard_decompositions(
    counts: &[u8; TILE_KIND_COUNT],
    groups: usize,
) -> Vec<(u8, Vec<(u8, GroupKind)>)> {
    let mut result = Vec::new();
    for pair in 0..TILE_KIND_COUNT {
        if counts[pair] < 2 {
            continue;
        }
        let mut remaining = *counts;
        remaining[pair] -= 2;
        let mut current = Vec::with_capacity(groups);
        let mut found = Vec::new();
        find_groups(&mut remaining, groups, &mut current, &mut found);
        for decomposition in found {
            result.push((pair as u8, decomposition));
        }
    }
    result
}

fn find_groups(
    counts: &mut [u8; TILE_KIND_COUNT],
    groups: usize,
    current: &mut Vec<(u8, GroupKind)>,
    found: &mut Vec<Vec<(u8, GroupKind)>>,
) {
    let Some(first) = counts.iter().position(|&count| count > 0) else {
        if groups == 0 {
            found.push(current.clone());
        }
        return;
    };
    if groups == 0 {
        return;
    }
    if counts[first] >= 3 {
        counts[first] -= 3;
        current.push((first as u8, GroupKind::Triplet));
        find_groups(counts, groups - 1, current, found);
        current.pop();
        counts[first] += 3;
    }
    let kind = first as u8;
    if is_suited(kind) && rank(kind) <= 7 && counts[first + 1] > 0 && counts[first + 2] > 0 {
        for offset in 0..3 {
            counts[first + offset] -= 1;
        }
        current.push((kind, GroupKind::Sequence));
        find_groups(counts, groups - 1, current, found);
        current.pop();
        for offset in 0..3 {
            counts[first + offset] += 1;
        }
    }
}

fn is_seven_pairs(counts: &[u8; TILE_KIND_COUNT]) -> bool {
    counts.iter().filter(|&&count| count == 2).count() == 7
}

fn is_thirteen_orphans(counts: &[u8; TILE_KIND_COUNT]) -> bool {
    KOKUSHI_KINDS
        .iter()
        .all(|&kind| counts[usize::from(kind)] >= 1)
        && counts.iter().sum::<u8>() == 14
        && KOKUSHI_KINDS
            .iter()
            .filter(|&&kind| counts[usize::from(kind)] == 2)
            .count()
            == 1
}

fn sequence_wait(start: u8, winning: u8) -> WaitType {
    let start_rank = rank(start);
    let winning_rank = rank(winning);
    if winning_rank == start_rank + 1 {
        WaitType::Kanchan
    } else if (start_rank, winning_rank) == (1, 3) || (start_rank, winning_rank) == (7, 7) {
        WaitType::Penchan
    } else {
        WaitType::Ryanmen
    }
}

/// Facts shared by every interpretation of one input.
struct HandFacts<'a> {
    input: &'a WinInput,
    rules: &'a Rules,
    menzen: bool,
    /// All 14+ owned tile kinds (concealed, winning tile, meld tiles).
    owned: Vec<TileInput>,
    /// Concealed tiles including the winning tile.
    concealed_counts: [u8; TILE_KIND_COUNT],
    meld_groups: Vec<Group>,
    is_dealer: bool,
}

impl HandFacts<'_> {
    fn all_kinds(&self) -> impl Iterator<Item = u8> + '_ {
        self.owned.iter().map(|tile| tile.kind)
    }

    fn situational_yaku(&self, yaku: &mut Vec<Yaku>) {
        let input = self.input;
        if input.method == WinMethod::Tsumo && self.menzen {
            yaku.push(Yaku::MenzenTsumo);
        }
        match input.riichi {
            RiichiStatus::DoubleRiichi => yaku.push(Yaku::DoubleRiichi),
            RiichiStatus::Riichi => yaku.push(Yaku::Riichi),
            RiichiStatus::None => {}
        }
        if input.ippatsu {
            yaku.push(Yaku::Ippatsu);
        }
        match input.situation {
            WinSituation::Chankan => yaku.push(Yaku::Chankan),
            WinSituation::Rinshan => yaku.push(Yaku::RinshanKaihou),
            WinSituation::Haitei => yaku.push(Yaku::Haitei),
            WinSituation::Houtei => yaku.push(Yaku::Houtei),
            _ => {}
        }
    }

    fn situational_yakuman(&self, yaku: &mut Vec<Yaku>) {
        match self.input.situation {
            WinSituation::Tenhou => yaku.push(Yaku::Tenhou),
            WinSituation::Chiihou => yaku.push(Yaku::Chiihou),
            _ => {}
        }
    }

    fn tile_set_yakuman(&self, yaku: &mut Vec<Yaku>) {
        if self.all_kinds().all(|kind| !is_suited(kind)) {
            yaku.push(Yaku::Tsuuiisou);
        }
        if self.all_kinds().all(|kind| GREEN_KINDS.contains(&kind)) {
            yaku.push(Yaku::Ryuuiisou);
        }
        if self.all_kinds().all(is_terminal) {
            yaku.push(Yaku::Chinroutou);
        }
    }

    fn tile_set_yaku(&self, yaku: &mut Vec<Yaku>) {
        if self.all_kinds().all(is_simple) && (self.menzen || self.rules.kuitan_enabled) {
            yaku.push(Yaku::Tanyao);
        }
        if self.all_kinds().all(is_terminal_or_honor) {
            yaku.push(Yaku::Honroutou);
        }
        let mut suits = self.all_kinds().filter(|&kind| is_suited(kind)).map(suit);
        if let Some(first) = suits.next() {
            if suits.all(|other| other == first) {
                let has_honor = self.all_kinds().any(|kind| !is_suited(kind));
                yaku.push(if has_honor {
                    Yaku::Honitsu
                } else {
                    Yaku::Chinitsu
                });
            }
        }
    }

    fn chuuren(&self) -> Option<Yaku> {
        if !self.input.melds.is_empty() {
            return None;
        }
        let kinds: Vec<u8> = self.all_kinds().collect();
        let first = kinds[0];
        if !is_suited(first)
            || kinds
                .iter()
                .any(|&kind| !is_suited(kind) || suit(kind) != suit(first))
        {
            return None;
        }
        let base = suit(first) * 9;
        let mut counts = [0u8; 9];
        for kind in kinds {
            counts[usize::from(kind - base)] += 1;
        }
        if counts
            .iter()
            .zip(CHUUREN_BASE_COUNTS.iter())
            .any(|(count, required)| count < required)
        {
            return None;
        }
        counts[usize::from(self.input.winning_tile.kind - base)] -= 1;
        Some(if counts == CHUUREN_BASE_COUNTS {
            Yaku::JunseiChuurenPoutou
        } else {
            Yaku::ChuurenPoutou
        })
    }

    fn pair_is_value(&self, pair: u8) -> bool {
        pair >= FIRST_DRAGON
            || pair == wind_kind(self.input.seat_wind)
            || pair == wind_kind(self.input.prevailing_wind)
    }

    fn standard_yakuman(&self, groups: &[Group], pair: u8, wait: WaitType, yaku: &mut Vec<Yaku>) {
        self.tile_set_yakuman(yaku);
        let triplets: Vec<u8> = groups
            .iter()
            .filter(|group| group.kind != GroupKind::Sequence)
            .map(|group| group.tile)
            .collect();
        if (FIRST_DRAGON..=33).all(|kind| triplets.contains(&kind)) {
            yaku.push(Yaku::Daisangen);
        }
        if concealed_triplet_count(groups) == 4 {
            yaku.push(if wait == WaitType::Tanki {
                Yaku::SuuankouTanki
            } else {
                Yaku::Suuankou
            });
        }
        let wind_triplets = (FIRST_HONOR..FIRST_DRAGON)
            .filter(|kind| triplets.contains(kind))
            .count();
        let pair_is_wind = (FIRST_HONOR..FIRST_DRAGON).contains(&pair);
        if wind_triplets == 4 {
            yaku.push(Yaku::Daisuushii);
        } else if wind_triplets == 3 && pair_is_wind && !triplets.contains(&pair) {
            yaku.push(Yaku::Shousuushii);
        }
        if groups
            .iter()
            .filter(|group| group.kind == GroupKind::Quad)
            .count()
            == 4
        {
            yaku.push(Yaku::Suukantsu);
        }
        if let Some(chuuren) = self.chuuren() {
            yaku.push(chuuren);
        }
    }

    fn normal_standard_yaku(
        &self,
        groups: &[Group],
        pair: u8,
        wait: WaitType,
        yaku: &mut Vec<Yaku>,
    ) {
        self.tile_set_yaku(yaku);
        let triplets: Vec<u8> = groups
            .iter()
            .filter(|group| group.kind != GroupKind::Sequence)
            .map(|group| group.tile)
            .collect();
        let sequences: Vec<u8> = groups
            .iter()
            .filter(|group| group.kind == GroupKind::Sequence)
            .map(|group| group.tile)
            .collect();

        let value_triplets = [
            (wind_kind(self.input.seat_wind), Yaku::SeatWind),
            (wind_kind(self.input.prevailing_wind), Yaku::PrevailingWind),
            (31, Yaku::WhiteDragon),
            (32, Yaku::GreenDragon),
            (33, Yaku::RedDragon),
        ];
        for (kind, value_yaku) in value_triplets {
            if triplets.contains(&kind) {
                yaku.push(value_yaku);
            }
        }

        if self.menzen
            && sequences.len() == groups.len()
            && !self.pair_is_value(pair)
            && wait == WaitType::Ryanmen
        {
            yaku.push(Yaku::Pinfu);
        }

        if self.menzen {
            let mut sequence_counts = [0u8; TILE_KIND_COUNT];
            for &start in &sequences {
                sequence_counts[usize::from(start)] += 1;
            }
            let identical_pairs: u8 = sequence_counts.iter().map(|count| count / 2).sum();
            if sequences.len() == groups.len() && identical_pairs == 2 {
                yaku.push(Yaku::Ryanpeikou);
            } else if identical_pairs > 0 {
                yaku.push(Yaku::Iipeikou);
            }
        }

        let has_sequence = !sequences.is_empty();
        let has_honor = self.all_kinds().any(|kind| !is_suited(kind));
        let group_has = |group: &Group, terminal_only: bool| -> bool {
            if group.kind == GroupKind::Sequence {
                matches!(rank(group.tile), 1 | 7)
            } else if terminal_only {
                is_terminal(group.tile)
            } else {
                is_terminal_or_honor(group.tile)
            }
        };
        if has_sequence
            && !has_honor
            && groups.iter().all(|group| group_has(group, true))
            && is_terminal(pair)
        {
            yaku.push(Yaku::Junchan);
        } else if has_sequence
            && has_honor
            && groups.iter().all(|group| group_has(group, false))
            && is_terminal_or_honor(pair)
        {
            yaku.push(Yaku::Chanta);
        }

        if (0..3).any(|s| {
            [0, 3, 6]
                .iter()
                .all(|offset| sequences.contains(&(s * 9 + offset)))
        }) {
            yaku.push(Yaku::Ittsuu);
        }
        if (0..7).any(|r| (0..3).all(|s| sequences.contains(&(s * 9 + r)))) {
            yaku.push(Yaku::SanshokuDoujun);
        }
        if (0..9).any(|r| (0..3).all(|s| triplets.contains(&(s * 9 + r)))) {
            yaku.push(Yaku::SanshokuDoukou);
        }
        if groups
            .iter()
            .filter(|group| group.kind == GroupKind::Quad)
            .count()
            == 3
        {
            yaku.push(Yaku::Sankantsu);
        }
        if !has_sequence {
            yaku.push(Yaku::Toitoi);
        }
        if concealed_triplet_count(groups) >= 3 {
            yaku.push(Yaku::Sanankou);
        }
        let dragon_triplets = (FIRST_DRAGON..=33)
            .filter(|kind| triplets.contains(kind))
            .count();
        if dragon_triplets == 2 && pair >= FIRST_DRAGON && !triplets.contains(&pair) {
            yaku.push(Yaku::Shousangen);
        }
    }

    fn standard_fu(
        &self,
        groups: &[Group],
        pair: u8,
        wait: WaitType,
    ) -> (u32, Vec<(FuReason, u32)>) {
        let input = self.input;
        let pinfu_shape = self.menzen
            && groups.iter().all(|group| group.kind == GroupKind::Sequence)
            && !self.pair_is_value(pair)
            && wait == WaitType::Ryanmen;
        let mut components = vec![(FuReason::Base, 20)];
        if input.method == WinMethod::Ron && self.menzen {
            components.push((FuReason::MenzenRon, 10));
        } else if input.method == WinMethod::Tsumo && !pinfu_shape {
            components.push((FuReason::Tsumo, 2));
        }

        let seat = pair == wind_kind(input.seat_wind);
        let prevailing = pair == wind_kind(input.prevailing_wind);
        if pair >= FIRST_DRAGON {
            components.push((FuReason::DragonPair, 2));
        } else if seat && prevailing {
            components.push((
                FuReason::DoubleWindPair,
                u32::from(self.rules.double_wind_pair_fu),
            ));
        } else {
            if seat {
                components.push((FuReason::SeatWindPair, 2));
            }
            if prevailing {
                components.push((FuReason::PrevailingWindPair, 2));
            }
        }

        for group in groups {
            let honor = is_terminal_or_honor(group.tile);
            let open = !group.is_concealed_for_scoring();
            let component = match (group.kind, open, honor) {
                (GroupKind::Sequence, _, _) => continue,
                (GroupKind::Triplet, true, false) => (FuReason::OpenSimpleTriplet, 2),
                (GroupKind::Triplet, true, true) => (FuReason::OpenTerminalOrHonorTriplet, 4),
                (GroupKind::Triplet, false, false) => (FuReason::ClosedSimpleTriplet, 4),
                (GroupKind::Triplet, false, true) => (FuReason::ClosedTerminalOrHonorTriplet, 8),
                (GroupKind::Quad, true, false) => (FuReason::OpenSimpleQuad, 8),
                (GroupKind::Quad, true, true) => (FuReason::OpenTerminalOrHonorQuad, 16),
                (GroupKind::Quad, false, false) => (FuReason::ClosedSimpleQuad, 16),
                (GroupKind::Quad, false, true) => (FuReason::ClosedTerminalOrHonorQuad, 32),
            };
            components.push(component);
        }
        match wait {
            WaitType::Kanchan => components.push((FuReason::KanchanWait, 2)),
            WaitType::Penchan => components.push((FuReason::PenchanWait, 2)),
            WaitType::Tanki => components.push((FuReason::TankiWait, 2)),
            _ => {}
        }
        let raw: u32 = components.iter().map(|(_, fu)| fu).sum();
        let rounded = if pinfu_shape && input.method == WinMethod::Tsumo {
            20
        } else {
            30.max(raw.div_ceil(10) * 10)
        };
        (rounded, components)
    }

    fn dora_count(&self) -> DoraCount {
        let count_for = |indicators: &[TileInput]| -> u8 {
            indicators
                .iter()
                .map(|indicator| {
                    let target = dora_kind(indicator.kind);
                    self.owned.iter().filter(|tile| tile.kind == target).count() as u8
                })
                .sum()
        };
        let red = if self.rules.red_dora_enabled {
            self.owned.iter().filter(|tile| tile.red).count() as u8
        } else {
            0
        };
        let ura = match &self.input.ura_dora {
            UraDora::Excluded => None,
            UraDora::Indicators(tiles) => Some(if self.input.riichi == RiichiStatus::None {
                0
            } else {
                count_for(tiles)
            }),
        };
        DoraCount {
            dora: count_for(&self.input.dora_indicators),
            red,
            ura,
        }
    }

    /// Build the scored candidate for one interpretation, or `None` if no yaku.
    #[allow(clippy::too_many_arguments)]
    fn candidate(
        &self,
        shape: WinningShape,
        wait: WaitType,
        pair: Option<u8>,
        groups: Vec<Group>,
        completed_group: Option<usize>,
        concealed_key: Vec<(u8, GroupKind)>,
        mut yakuman: Vec<Yaku>,
        mut normal: Vec<Yaku>,
        fu: Option<(u32, Vec<(FuReason, u32)>)>,
    ) -> Option<Score> {
        let rules = self.rules;
        let tie_key = TieKey {
            shape,
            pair,
            concealed: concealed_key,
            completed: completed_group,
        };
        let ura_dora_excluded = self.input.ura_dora == UraDora::Excluded;
        let (yaku, dora, yaku_han, han, fu, fu_components, yakuman_units) = if !yakuman.is_empty() {
            yakuman.sort_unstable();
            yakuman.dedup();
            let yaku: Vec<YakuValue> = yakuman
                .into_iter()
                .map(|yaku| YakuValue {
                    yaku,
                    han: 0,
                    yakuman_units: if rules.double_yakuman_variants.contains(&yaku) {
                        2
                    } else {
                        1
                    },
                })
                .collect();
            let units: u32 = yaku
                .iter()
                .map(|value| u32::from(value.yakuman_units))
                .sum();
            (yaku, None, 0, 0, None, Vec::new(), units)
        } else {
            normal.sort_unstable();
            normal.dedup();
            let yaku: Vec<YakuValue> = normal
                .into_iter()
                .filter_map(|yaku| {
                    yaku.han(self.menzen).map(|han| YakuValue {
                        yaku,
                        han,
                        yakuman_units: 0,
                    })
                })
                .collect();
            if yaku.is_empty() {
                return None;
            }
            let yaku_han: u32 = yaku.iter().map(|value| u32::from(value.han)).sum();
            let dora = self.dora_count();
            let (fu, components) = fu.expect("non-yakuman candidates carry fu");
            (
                yaku,
                Some(dora),
                yaku_han,
                yaku_han + dora.total(),
                Some(fu),
                components,
                0,
            )
        };

        let (base_points, limit) = base_points_and_limit(han, fu, yakuman_units, rules);
        let round_up = |points: u32| points.div_ceil(100) * 100;
        let (ron_payment, tsumo_dealer_payment, tsumo_non_dealer_payment, winner_points) =
            match (self.input.method, self.is_dealer) {
                (WinMethod::Ron, dealer) => {
                    let payment = round_up(base_points * if dealer { 6 } else { 4 });
                    (Some(payment), None, None, payment)
                }
                (WinMethod::Tsumo, true) => {
                    let each = round_up(base_points * 2);
                    (None, None, Some(each), each * 3)
                }
                (WinMethod::Tsumo, false) => {
                    let dealer = round_up(base_points * 2);
                    let non_dealer = round_up(base_points);
                    (
                        None,
                        Some(dealer),
                        Some(non_dealer),
                        dealer + 2 * non_dealer,
                    )
                }
            };
        Some(Score {
            shape,
            wait,
            pair,
            groups,
            completed_group,
            yaku,
            dora,
            ura_dora_excluded,
            yaku_han,
            han,
            fu,
            fu_components,
            yakuman_units,
            is_dealer: self.is_dealer,
            method: self.input.method,
            base_points,
            limit,
            ron_payment,
            tsumo_dealer_payment,
            tsumo_non_dealer_payment,
            winner_points,
            tie_key,
        })
    }
}

fn concealed_triplet_count(groups: &[Group]) -> usize {
    groups
        .iter()
        .filter(|group| group.kind != GroupKind::Sequence && group.is_concealed_for_scoring())
        .count()
}

fn base_points_and_limit(
    han: u32,
    fu: Option<u32>,
    yakuman_units: u32,
    rules: &Rules,
) -> (u32, ScoreLimit) {
    if yakuman_units > 0 {
        let units = if rules.multiple_yakuman_enabled {
            yakuman_units
        } else {
            1
        };
        return (8_000 * units, ScoreLimit::Yakuman);
    }
    let fu = fu.unwrap_or(0);
    if han >= 13 {
        return if rules.counted_yakuman_enabled {
            (8_000, ScoreLimit::Yakuman)
        } else {
            (6_000, ScoreLimit::Sanbaiman)
        };
    }
    if han >= 11 {
        return (6_000, ScoreLimit::Sanbaiman);
    }
    if han >= 8 {
        return (4_000, ScoreLimit::Baiman);
    }
    if han >= 6 {
        return (3_000, ScoreLimit::Haneman);
    }
    if han >= 5 || (han == 4 && fu >= 40) || (han == 3 && fu >= 70) {
        return (2_000, ScoreLimit::Mangan);
    }
    if rules.rounded_mangan_enabled && ((han == 4 && fu == 30) || (han == 3 && fu == 60)) {
        return (2_000, ScoreLimit::Mangan);
    }
    (fu * 2u32.pow(han + 2), ScoreLimit::None)
}

/// Order used to pick the best candidate: higher winner points, then more
/// yakuman units, more han, more fu; remaining ties take the smallest
/// interpretation key (shape standard < seven pairs < thirteen orphans, then
/// pair kind, concealed groups in canonical order, winning-tile position with
/// the pair first).
fn compare_candidates(left: &Score, right: &Score) -> Ordering {
    right
        .winner_points
        .cmp(&left.winner_points)
        .then(right.yakuman_units.cmp(&left.yakuman_units))
        .then(right.han.cmp(&left.han))
        .then(right.fu.unwrap_or(0).cmp(&left.fu.unwrap_or(0)))
        .then_with(|| left.tie_key.cmp(&right.tie_key))
}

/// Evaluate one concrete hand.
///
/// Every interpretation (shape x decomposition x winning-tile assignment) is
/// scored; the candidate ordered first by [`compare_candidates`] is returned.
/// Invalid input is an `Err`; a non-complete hand and a complete hand without
/// yaku are normal results.
pub fn evaluate(input: &WinInput, rules: &Rules) -> Result<Evaluation, String> {
    validate(input, rules)?;

    let mut concealed_counts = [0u8; TILE_KIND_COUNT];
    for tile in input
        .concealed
        .iter()
        .chain(std::iter::once(&input.winning_tile))
    {
        concealed_counts[usize::from(tile.kind)] += 1;
    }
    let owned: Vec<TileInput> = input
        .concealed
        .iter()
        .chain(std::iter::once(&input.winning_tile))
        .chain(input.melds.iter().flat_map(|meld| meld.tiles.iter()))
        .copied()
        .collect();
    let facts = HandFacts {
        input,
        rules,
        menzen: input.melds.iter().all(|meld| meld.kind == MeldKind::Ankan),
        owned,
        concealed_counts,
        meld_groups: input.melds.iter().map(meld_group).collect(),
        is_dealer: input.seat_wind == 0,
    };

    let winning = input.winning_tile.kind;
    let mut complete = false;
    let mut candidates: Vec<Score> = Vec::new();

    for (pair, concealed) in
        standard_decompositions(&facts.concealed_counts, MAX_MELDS - input.melds.len())
    {
        complete = true;
        // Winning-tile assignments: the pair, then each distinct concealed
        // group containing the winning tile kind.
        let mut assignments: Vec<Option<usize>> = Vec::new();
        if pair == winning {
            assignments.push(None);
        }
        for (index, &(tile, kind)) in concealed.iter().enumerate() {
            let contains = match kind {
                GroupKind::Sequence => (tile..tile + 3).contains(&winning),
                _ => tile == winning,
            };
            if contains && !concealed[..index].contains(&(tile, kind)) {
                assignments.push(Some(index));
            }
        }
        for completed in assignments {
            let wait = match completed {
                None => WaitType::Tanki,
                Some(index) => match concealed[index].1 {
                    GroupKind::Sequence => sequence_wait(concealed[index].0, winning),
                    _ => WaitType::Shanpon,
                },
            };
            let mut groups: Vec<Group> = concealed
                .iter()
                .enumerate()
                .map(|(index, &(tile, kind))| Group {
                    kind,
                    tile,
                    is_open: false,
                    completed_by_ron: input.method == WinMethod::Ron
                        && kind == GroupKind::Triplet
                        && completed == Some(index),
                })
                .collect();
            groups.extend(facts.meld_groups.iter().copied());

            let mut yakuman = Vec::new();
            facts.situational_yakuman(&mut yakuman);
            facts.standard_yakuman(&groups, pair, wait, &mut yakuman);
            let mut normal = Vec::new();
            let mut fu = None;
            if yakuman.is_empty() {
                facts.situational_yaku(&mut normal);
                facts.normal_standard_yaku(&groups, pair, wait, &mut normal);
                fu = Some(facts.standard_fu(&groups, pair, wait));
            }
            if let Some(score) = facts.candidate(
                WinningShape::Standard,
                wait,
                Some(pair),
                groups,
                completed,
                concealed.clone(),
                yakuman,
                normal,
                fu,
            ) {
                candidates.push(score);
            }
        }
    }

    if input.melds.is_empty() && is_seven_pairs(&facts.concealed_counts) {
        complete = true;
        let mut yakuman = Vec::new();
        facts.situational_yakuman(&mut yakuman);
        facts.tile_set_yakuman(&mut yakuman);
        let mut normal = Vec::new();
        let mut fu = None;
        if yakuman.is_empty() {
            facts.situational_yaku(&mut normal);
            facts.tile_set_yaku(&mut normal);
            normal.push(Yaku::Chiitoitsu);
            fu = Some((25, vec![(FuReason::SevenPairs, 25)]));
        }
        if let Some(score) = facts.candidate(
            WinningShape::SevenPairs,
            WaitType::Tanki,
            None,
            Vec::new(),
            None,
            Vec::new(),
            yakuman,
            normal,
            fu,
        ) {
            candidates.push(score);
        }
    }

    if input.melds.is_empty() && is_thirteen_orphans(&facts.concealed_counts) {
        complete = true;
        let mut before = facts.concealed_counts;
        before[usize::from(winning)] -= 1;
        let thirteen_sided = KOKUSHI_KINDS
            .iter()
            .all(|&kind| before[usize::from(kind)] == 1);
        let mut yakuman = Vec::new();
        facts.situational_yakuman(&mut yakuman);
        yakuman.push(if thirteen_sided {
            Yaku::KokushiMusou13Wait
        } else {
            Yaku::KokushiMusou
        });
        let wait = if thirteen_sided {
            WaitType::KokushiThirteenSided
        } else {
            WaitType::KokushiSingle
        };
        if let Some(score) = facts.candidate(
            WinningShape::ThirteenOrphans,
            wait,
            None,
            Vec::new(),
            None,
            Vec::new(),
            yakuman,
            Vec::new(),
            None,
        ) {
            candidates.push(score);
        }
    }

    if !complete {
        return Ok(Evaluation::NotComplete);
    }
    candidates.sort_by(compare_candidates);
    match candidates.into_iter().next() {
        Some(best) => Ok(Evaluation::Scored(Box::new(best))),
        None => Ok(Evaluation::CompleteWithoutYaku),
    }
}

#[cfg(test)]
mod tests;
