"""Issue #211: 固定upstreamに対するbounded differential check。

`tools/kobalab_0004_reference/generate_upstream_fixture.js`が、固定した
kobalab/majiang-ai（commit e75a9720a12b84c03e6c61c3960c1844b8982eb4）と
@kobalab/majiang-core 1.4.1を実行して書き出した`tests/fixtures/
kobalab_0004_upstream.json`を読み、lisjong側の中間値と最終decisionを照合する。

fixtureはupstreamのJS表現（majiang牌譜表記）を言語非依存のJSONとして保持し、
このtest側で`PolicyInput` / `InternalAction`へ変換する。upstreamのobject modelを
lisjongの公開contractへ持ち込まない。Node.jsはCIで実行しない。

fixtureは必須であり、欠落時はskipせず失敗する。自手に同一牌種4枚を持つ手牌も
含め、全turnの最終decision（暗槓・打牌・立直）を照合する。

中間値の既知差分は`KNOWN_CANDIDATE_DIFFERENCES`に、fixture id・打牌候補・両側の
期待値・理由を列挙したものだけに限定する。列挙した差分が再現しない場合も失敗する。
現fixtureで観測された差分は、majiang-coreの向聴数式が同一牌種5枚目を要する単騎を
聴牌として数え、lisjong exact shantenが数えないことに起因する（向聴定義差）。
Policyはexact shantenを使う参照実装として位置付けるため、この差は設計上許容する。
ただし観測値を固定して変化を検出し、新しい不一致は既知差分へ機械的に追加せず
個別に原因を調査する。

- 表現差: sourceの`get_dapai()`は、ツモ牌が唯一の通常5で赤5も持つ場合に
  実在しない手出し通常5を列挙する。lisjongの合法手には現れないため変換時に除外する。
"""

import json
import unittest
from collections import Counter
from pathlib import Path

from lisjong.belief import derive_remaining_tile_inventory, tile_type_index
from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies import Kobalab0004ReferencePolicy
from lisjong.policies.kobalab_0004_reference import (
    _improving_tile_types,
    _PublicCounts,
    evaluation_order,
)
from lisjong.policy_contract.action import (
    AnkanAction,
    DiscardAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    PassAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.discard import Discard
from lisjong.policy_contract.meld import MeldKind, PublicMeld
from lisjong.policy_contract.own_hand_state import OwnHandState
from lisjong.policy_contract.player_state import PlayerPublicState
from lisjong.policy_contract.policy_execution import execute_policy
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.riichi import RiichiState
from lisjong.policy_contract.round_state import RoundState
from lisjong.policy_contract.seat import Seat
from lisjong.policy_contract.tile import Tile, TileCategory, TileType
from lisjong.policy_contract.wind import Wind

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "kobalab_0004_upstream.json"

_SUITS = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}
_WINDS = (Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH)
_MARK_OFFSET = {"+": 1, "=": 2, "-": 3}
"""majiang表記の方向記号: 鳴いた側から見た放銃者の相対位置（+下家 / =対面 / -上家）。"""


def _tile(text: str) -> Tile:
    number = int(text[1])
    return Tile(TileType(_SUITS[text[0]], number or 5), is_red=number == 0)


def _river(entries: list[str], owner: int) -> tuple[Discard, ...]:
    discards = []
    for index, entry in enumerate(entries):
        called_by = None
        if entry[-1] in _MARK_OFFSET:
            called_by = Seat((owner - _MARK_OFFSET[entry[-1]]) % 4)
        discards.append(
            Discard(
                tile=_tile(entry[:2]),
                tsumogiri="_" in entry,
                order=index * 4 + owner,
                called_by=called_by,
            )
        )
    return tuple(discards)


def _meld(text: str, owner: int) -> PublicMeld:
    suit = text[0]
    body = text[1:]
    digits = [c for c in body if c.isdigit()]
    tiles = tuple(_tile(suit + d) for d in digits)
    marks = [(i, c) for i, c in enumerate(body) if c in _MARK_OFFSET]
    if not marks:
        return PublicMeld(
            kind=MeldKind.ANKAN, tiles=tiles, from_seat=None, called_tile=None
        )
    position, mark = marks[0]
    called = _tile(suit + body[position - 1])
    from_seat = Seat((owner + _MARK_OFFSET[mark]) % 4)
    if len(digits) == 4:
        kind = MeldKind.DAIMINKAN if position == len(body) - 1 else MeldKind.KAKAN
    elif len({tile.tile_type for tile in tiles}) == 1:
        kind = MeldKind.PON
    else:
        kind = MeldKind.CHI
    return PublicMeld(kind=kind, tiles=tiles, from_seat=from_seat, called_tile=called)


def _policy_input(
    state: dict, *, self_riichi: RiichiState | None = None
) -> PolicyInput:
    menfeng = state["menfeng"]
    players = []
    for seat in range(4):
        riichi = RiichiState.ACCEPTED if state["lizhi"][seat] else RiichiState.NONE
        if seat == menfeng and self_riichi is not None:
            riichi = self_riichi
        players.append(
            PlayerPublicState(
                score=25000,
                discards=_river(state["he"][seat], seat),
                melds=tuple(_meld(m, seat) for m in state["fulou"][seat]),
                riichi=riichi,
            )
        )
    zimo = state["zimo"]
    drawn = _tile(zimo) if zimo and len(zimo) == 2 else None
    return PolicyInput(
        self_seat=Seat(menfeng),
        round=RoundState(
            round_wind=_WINDS[state["zhuangfeng"]],
            hand_number=1,
            dealer_seat=Seat.SEAT_0,
            honba=0,
            riichi_sticks=0,
            dora_indicators=tuple(_tile(p) for p in state["baopai"]),
            live_wall_tiles_remaining=state["paishu"],
        ),
        players=tuple(players),
        own_hand=OwnHandState(
            concealed_tiles=tuple(_tile(p) for p in state["concealed"]),
            drawn_tile=drawn,
        ),
    )


def _discard_action(text: str, state: dict) -> DiscardAction | None:
    """source打牌表記をDiscardActionへ。実在しない手出し候補はNoneを返す。"""
    actor = Seat(state["menfeng"])
    tile = _tile(text[:2])
    tsumogiri = "_" in text
    if not tsumogiri:
        concealed = [_tile(p) for p in state["concealed"]]
        zimo = state["zimo"]
        spare = concealed.count(tile) - (
            1 if zimo and len(zimo) == 2 and _tile(zimo) == tile else 0
        )
        if spare <= 0:
            return None
    return DiscardAction(actor=actor, tile=tile, tsumogiri=tsumogiri)


def _gang_action(text: str, state: dict) -> AnkanAction | KakanAction:
    actor = Seat(state["menfeng"])
    meld = _meld(text, state["menfeng"])
    if meld.kind is MeldKind.ANKAN:
        return AnkanAction(actor=actor, tiles=meld.tiles)
    return KakanAction(
        actor=actor,
        added_tile=_tile(text[0] + text[-1]),
        from_seat=meld.from_seat,
        called_tile=meld.called_tile,
    )


def _discards(texts: list[str], state: dict) -> tuple[DiscardAction, ...]:
    actions: list[DiscardAction] = []
    for text in texts:
        action = _discard_action(text.rstrip("*"), state)
        if action is not None and action not in actions:
            actions.append(action)
    return tuple(actions)


def _legal_actions(record: dict) -> tuple[object, ...]:
    state, legal = record["state"], record["legal"]
    actor = Seat(state["menfeng"])
    actions: list[object] = []
    if legal["hule"]:
        actions.append(TsumoAction(actor=actor, winning_tile=_tile(state["zimo"])))
    if legal["pingju"]:
        actions.append(KyuushuKyuuhaiAction(actor=actor))
    actions.extend(_gang_action(m, state) for m in legal["gang"])
    if legal["lizhi"]:
        actions.append(RiichiAction(actor=actor))
    actions.extend(_discards(legal["dapai"], state))
    return tuple(actions)


KNOWN_CANDIDATE_DIFFERENCES = {
    ("kan-order", "z2_"): {
        "upstream": {"xiangting": 1, "ev": 26, "extra_tingpai": ("z1",)},
        "lisjong": {"xiangting": 1, "ev": 24},
        "reason": (
            "打牌後m1111p0555s234z11にz1を加えたm111+p555+s234+z111は、残りm1/p5の"
            "単騎がいずれも5枚目を要する。majiang-coreは聴牌と数えてz1を改善牌に"
            "含め（残り2枚）、lisjongは数えない。"
        ),
    },
    ("red-five-tie", "p1"): {
        "upstream": {"xiangting": 0, "ev": 0, "extra_tingpai": ()},
        "lisjong": {"xiangting": 1, "ev": 121},
        "reason": (
            "打牌後m123456789s0555はs5単騎（5枚目）待ちのみ。majiang-coreは和了牌の"
            "無い聴牌として向聴0・改善牌なしとし、lisjongは1向聴として扱う。"
        ),
    },
}
"""中間値の既知差分: (fixture id, source打牌表記) -> 両側の期待値と理由。

いずれも同一牌種5枚目を要する分解に関する向聴定義差で、設計上許容する。
どちらのturnでも最終decisionは暗槓であり、upstreamと一致する。
"""


def _load() -> dict:
    if not FIXTURE_PATH.exists():
        raise AssertionError(
            f"required upstream differential fixture is missing: {FIXTURE_PATH}; "
            "see tools/kobalab_0004_reference/generate_upstream_fixture.js"
        )
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class UpstreamDifferentialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _load()

    def _turns(self) -> list[dict]:
        return self.fixture["games"] + self.fixture["crafted_turns"]

    def test_pinned_upstream_identity(self) -> None:
        upstream = self.fixture["upstream"]
        self.assertEqual(
            upstream["majiang_ai"]["commit"],
            "e75a9720a12b84c03e6c61c3960c1844b8982eb4",
        )
        self.assertEqual(upstream["majiang_core"]["version"], "1.4.1")

    def test_shanten_and_improving_tiles(self) -> None:
        for case in self.fixture["shanten"]:
            tiles = [_tile(p) for p in case["tiles"]]
            with self.subTest(tiles=case["tiles"]):
                self.assertEqual(calculate_shanten(tiles), case["xiangting"])
                if case["tingpai"] is not None:
                    self.assertEqual(
                        set(_improving_tile_types(tiles)),
                        {_tile(p).tile_type for p in case["tingpai"]},
                    )

    def test_remaining_counts_match_suanpai(self) -> None:
        for record in self._turns():
            with self.subTest(id=record["id"]):
                conservation = derive_remaining_tile_inventory(
                    _policy_input(record["state"])
                )
                paishu = record["suanpai_paishu"]
                for suit, category in _SUITS.items():
                    for rank in range(1, len(paishu[suit])):
                        index = tile_type_index(TileType(category, rank))
                        self.assertEqual(
                            conservation.remaining_tile_counts[index],
                            paishu[suit][rank],
                            f"{suit}{rank}",
                        )
                    if category is not TileCategory.HONOR:
                        red_index = "mps".index(suit)
                        self.assertEqual(
                            conservation.remaining_red_five_counts[red_index],
                            paishu[suit][0],
                        )

    def test_paijia_ukeire_and_evaluation_order(self) -> None:
        observed_differences = set()
        for record in self._turns():
            evaluation = record["evaluation"]
            if evaluation is None:
                continue
            state = record["state"]
            policy_input = _policy_input(state)
            concealed = list(policy_input.own_hand.concealed_tiles)
            with self.subTest(id=record["id"]):
                counts = _PublicCounts(policy_input)
                expected_order = []
                for candidate in evaluation["candidates"]:
                    action = _discard_action(candidate["p"], state)
                    self.assertEqual(
                        counts.paijia(_tile(candidate["p"][:2])), candidate["paijia"]
                    )
                    if action is None or action in expected_order:
                        continue
                    expected_order.append(action)
                    after = list(concealed)
                    after.remove(action.tile)
                    improving = _improving_tile_types(after)
                    ours = {
                        "xiangting": calculate_shanten(after),
                        "ev": sum(counts.remaining(t) for t in improving),
                    }
                    upstream_tingpai = {
                        _tile(p).tile_type for p in candidate["tingpai"]
                    }
                    key = (record["id"], candidate["p"])
                    known = KNOWN_CANDIDATE_DIFFERENCES.get(key)
                    if known is None:
                        self.assertEqual(ours["xiangting"], candidate["xiangting"], key)
                        self.assertEqual(ours["ev"], candidate["ev"], key)
                        self.assertEqual(set(improving), upstream_tingpai, key)
                        continue
                    observed_differences.add(key)
                    upstream = known["upstream"]
                    self.assertEqual(candidate["xiangting"], upstream["xiangting"], key)
                    self.assertEqual(candidate["ev"], upstream["ev"], key)
                    self.assertEqual(
                        upstream_tingpai - set(improving),
                        {_tile(p).tile_type for p in upstream["extra_tingpai"]},
                        key,
                    )
                    self.assertEqual(ours, known["lisjong"], key)
                self.assertEqual(
                    calculate_shanten(concealed), evaluation["n_xiangting"]
                )
                self.assertEqual(
                    list(evaluation_order(policy_input, tuple(expected_order))),
                    expected_order,
                )
        self.assertEqual(observed_differences, set(KNOWN_CANDIDATE_DIFFERENCES))

    def test_final_decisions(self) -> None:
        policy = Kobalab0004ReferencePolicy()
        for record in self._turns():
            state, decision = record["state"], record["decision"]
            actor = Seat(state["menfeng"])
            policy_input = _policy_input(state)
            context = DecisionContext(
                input=policy_input, legal_actions=_legal_actions(record)
            )
            with self.subTest(id=record["id"], decision=decision):
                chosen = execute_policy(policy, context)
                if "hule" in decision:
                    self.assertIsInstance(chosen, TsumoAction)
                elif "daopai" in decision:
                    self.assertIsInstance(chosen, KyuushuKyuuhaiAction)
                elif "gang" in decision:
                    self.assertEqual(chosen, _gang_action(decision["gang"], state))
                else:
                    dapai = decision["dapai"]
                    if dapai.endswith("*"):
                        self.assertEqual(chosen, RiichiAction(actor=actor))
                        declared = DecisionContext(
                            input=_policy_input(
                                state, self_riichi=RiichiState.DECLARED
                            ),
                            legal_actions=_discards(record["legal"]["lizhi"], state),
                        )
                        chosen = execute_policy(policy, declared)
                    self.assertEqual(chosen, _discard_action(dapai.rstrip("*"), state))

    def test_quad_holding_turns_cover_ankan_discard_and_riichi(self) -> None:
        """同一牌種4枚持ちturnが最終decision比較の対象に含まれることを固定する。"""
        kinds = set()
        for record in self._turns():
            counts = Counter(_tile(p).tile_type for p in record["state"]["concealed"])
            if max(counts.values()) < 4:
                continue
            decision = record["decision"]
            if "gang" in decision:
                kinds.add("ankan")
            elif "dapai" in decision:
                kinds.add("riichi" if decision["dapai"].endswith("*") else "discard")
        self.assertEqual(kinds, {"ankan", "discard", "riichi"})

    def test_chankan(self) -> None:
        policy = Kobalab0004ReferencePolicy()
        for record in self.fixture["crafted_chankan"]:
            state = dict(record["state"])
            target = record["gang"]["l"]
            fulou = [list(melds) for melds in state["fulou"]]
            fulou[target].append(record["gang"]["m"])
            state["fulou"] = fulou
            policy_input = _policy_input(state)
            actor = Seat(state["menfeng"])
            meld = _meld(record["gang"]["m"], target)
            ron = RonAction(
                actor=actor,
                target=Seat(target),
                winning_tile=Tile(meld.tiles[0].tile_type),
            )
            context = DecisionContext(
                input=policy_input, legal_actions=(ron, PassAction(actor=actor))
            )
            with self.subTest(id=record["id"]):
                chosen = execute_policy(policy, context)
                self.assertEqual(chosen == ron, record["hule"])


if __name__ == "__main__":
    unittest.main()
