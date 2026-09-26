"""Issue #211: 固定upstreamに対するbounded differential check。

`tools/kobalab_0004_reference/generate_upstream_fixture.js`が、固定した
kobalab/majiang-ai（commit e75a9720a12b84c03e6c61c3960c1844b8982eb4）と
@kobalab/majiang-core 1.4.1を実行して書き出した`tests/fixtures/
kobalab_0004_upstream.json`を読み、lisjong側の中間値と最終decisionを照合する。

fixtureはupstreamのJS表現（majiang牌譜表記）を言語非依存のJSONとして保持し、
このtest側で`PolicyInput` / `InternalAction`へ変換する。upstreamのobject modelを
lisjongの公開contractへ持ち込まない。Node.jsはCIで実行しない。

既知の分類済み差分:

- 向聴定義差（Policy evaluatorの差）: majiang-coreの式は同一牌種5枚目を要する
  分解を数えるが、lisjong exact shantenは数えない。自手に同一牌種4枚を持つ
  手牌でだけlisjongが+1になり得る。該当ケースは中間値の厳密比較から除外し、
  差がこの形に限られることを検証する。
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


def _has_quad(tiles: list[Tile]) -> bool:
    return max(Counter(tile.tile_type for tile in tiles).values(), default=0) >= 4


def _load() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@unittest.skipUnless(
    FIXTURE_PATH.exists(),
    "upstream differential fixture is not generated; see "
    "tools/kobalab_0004_reference/generate_upstream_fixture.js",
)
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
        known_difference = 0
        for case in self.fixture["shanten"]:
            tiles = [_tile(p) for p in case["tiles"]]
            with self.subTest(tiles=case["tiles"]):
                ours = calculate_shanten(tiles)
                if ours != case["xiangting"]:
                    # 5枚目を要する分解だけが許容される分類済み差分。
                    self.assertTrue(_has_quad(tiles))
                    self.assertEqual(ours, case["xiangting"] + 1)
                    known_difference += 1
                    continue
                if case["tingpai"] is not None and not _has_quad(tiles):
                    self.assertEqual(
                        set(_improving_tile_types(tiles)),
                        {_tile(p).tile_type for p in case["tingpai"]},
                    )
        self.assertLessEqual(known_difference * 10, len(self.fixture["shanten"]))

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
                    if _has_quad(concealed):
                        continue
                    after = list(concealed)
                    after.remove(action.tile)
                    self.assertEqual(calculate_shanten(after), candidate["xiangting"])
                    self.assertEqual(
                        sum(counts.remaining(t) for t in _improving_tile_types(after)),
                        candidate["ev"],
                    )
                if not _has_quad(concealed):
                    self.assertEqual(
                        calculate_shanten(concealed), evaluation["n_xiangting"]
                    )
                self.assertEqual(
                    list(evaluation_order(policy_input, tuple(expected_order))),
                    expected_order,
                )

    def test_final_decisions(self) -> None:
        policy = Kobalab0004ReferencePolicy()
        for record in self._turns():
            state, decision = record["state"], record["decision"]
            concealed = [_tile(p) for p in state["concealed"]]
            if _has_quad(concealed):
                continue  # 分類済み向聴定義差の影響を受け得る
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
