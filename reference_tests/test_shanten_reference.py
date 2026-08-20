import importlib.metadata
import random
import unittest
from collections.abc import Iterable

from mahjong.shanten import Shanten

from lisjong.hand_evaluation import calculate_shanten
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

_EXPECTED_REFERENCE_VERSION = "2.0.0"
_RANDOM_SEED = 700034
_SAMPLES_PER_HAND_SIZE = 500
_VALID_HAND_SIZES = (1, 2, 4, 5, 7, 8, 10, 11, 13, 14)

_CATEGORY_BY_SUFFIX = {
    "m": TileCategory.MANZU,
    "p": TileCategory.PINZU,
    "s": TileCategory.SOUZU,
    "z": TileCategory.HONOR,
}
_REFERENCE_CATEGORY_OFFSETS = {
    TileCategory.MANZU: 0,
    TileCategory.PINZU: 9,
    TileCategory.SOUZU: 18,
    TileCategory.HONOR: 27,
}


def _hand(notation: str) -> tuple[Tile, ...]:
    tiles: list[Tile] = []
    ranks: list[int] = []
    for character in notation:
        if character.isdigit():
            ranks.append(int(character))
            continue

        category = _CATEGORY_BY_SUFFIX[character]
        for rank in ranks:
            is_red = rank == 0
            tiles.append(Tile(TileType(category, 5 if is_red else rank), is_red=is_red))
        ranks = []

    if ranks:
        raise ValueError(f"notation has trailing ranks: {notation!r}")
    return tuple(tiles)


def _to_reference_tiles_34(tiles: Iterable[Tile]) -> list[int]:
    """Convert lisjong tiles without using the production shanten adapter."""
    counts = [0] * 34
    for tile in tiles:
        tile_type = tile.tile_type
        index = _REFERENCE_CATEGORY_OFFSETS[tile_type.category] + tile_type.rank - 1
        counts[index] += 1
    return counts


def _all_base_tile_types() -> tuple[TileType, ...]:
    suited = tuple(
        TileType(category, rank)
        for category in (
            TileCategory.MANZU,
            TileCategory.PINZU,
            TileCategory.SOUZU,
        )
        for rank in range(1, 10)
    )
    honors = tuple(TileType(TileCategory.HONOR, rank) for rank in range(1, 8))
    return suited + honors


_PHYSICAL_TILE_POOL = tuple(
    Tile(tile_type) for tile_type in _all_base_tile_types() for _ in range(4)
)


class ReferenceAdapterTest(unittest.TestCase):
    def test_maps_suits_and_honors_to_reference_indices(self) -> None:
        counts = _to_reference_tiles_34(_hand("19m19p19s17z"))

        self.assertEqual(
            [index for index, count in enumerate(counts) if count],
            [0, 8, 9, 17, 18, 26, 27, 33],
        )

    def test_normalizes_red_fives_to_the_base_tile_kind(self) -> None:
        counts = _to_reference_tiles_34(_hand("50m50p50s"))

        self.assertEqual(counts[4], 2)
        self.assertEqual(counts[13], 2)
        self.assertEqual(counts[22], 2)
        self.assertEqual(sum(counts), 6)


class CuratedShantenDifferentialTest(unittest.TestCase):
    CASES = (
        ("one tile", "1p", 0),
        ("pair", "11p", -1),
        ("four tiles", "123m1p", 0),
        ("five tiles", "123m11p", -1),
        ("seven tiles", "123456m1p", 0),
        ("eight tiles", "123456m11p", -1),
        ("ten tiles", "19m19p19s1234z", 6),
        ("eleven tiles", "123456789m11p", -1),
        ("thirteen tiles", "123456789m1122p", 0),
        ("fourteen tiles", "123456789m11122p", -1),
        ("positive shanten", "123456789m1p24s7z", 1),
        ("seven pairs", "11m22m33p44s55z66z77z", -1),
        ("thirteen orphans", "19m19p19s12345677z", -1),
        ("minimum is seven pairs", "1199m1199p1199s1z", 0),
        ("minimum is regular", "123456789m1122p", 0),
        ("overlapping sequences", "22334455m678p99s", 0),
        ("overlapping triplets and pair", "1111m222m", 0),
        ("four copies", "1111m", 1),
        ("impossible fifth copy", "1111p666z", 1),
        ("dead honor spares", "111122223333z1m", 3),
        ("red five", "123456789m11p340s", -1),
        ("four copies including red", "5550m", 1),
    )

    def test_curated_hands_match_expected_and_reference_results(self) -> None:
        self.assertEqual(
            {len(_hand(notation)) for _, notation, _ in self.CASES},
            set(_VALID_HAND_SIZES),
        )

        for description, notation, expected in self.CASES:
            tiles = _hand(notation)
            reference_tiles_34 = _to_reference_tiles_34(tiles)
            lisjong_result = calculate_shanten(tiles)
            reference_result = Shanten.calculate_shanten(reference_tiles_34)
            with self.subTest(description=description, notation=notation):
                self.assertEqual(lisjong_result, expected)
                self.assertEqual(reference_result, expected)


class SeededRandomShantenDifferentialTest(unittest.TestCase):
    def test_reference_package_version_is_pinned(self) -> None:
        self.assertEqual(
            importlib.metadata.version("mahjong"),
            _EXPECTED_REFERENCE_VERSION,
        )

    def test_seeded_physical_hands_match_reference(self) -> None:
        self.assertEqual(len(_PHYSICAL_TILE_POOL), 136)
        rng = random.Random(_RANDOM_SEED)
        reference_version = importlib.metadata.version("mahjong")

        for hand_size in _VALID_HAND_SIZES:
            for sample_index in range(_SAMPLES_PER_HAND_SIZE):
                tiles = tuple(rng.sample(_PHYSICAL_TILE_POOL, hand_size))
                reference_tiles_34 = _to_reference_tiles_34(tiles)
                lisjong_result = calculate_shanten(tiles)
                reference_result = Shanten.calculate_shanten(reference_tiles_34)
                diagnostic = (
                    f"original_hand={tiles!r}; hand_size={hand_size}; "
                    f"reference_tiles_34={reference_tiles_34!r}; "
                    f"lisjong_result={lisjong_result}; "
                    f"reference_result={reference_result}; "
                    f"reference_version={reference_version}; seed={_RANDOM_SEED}; "
                    f"sample_index={sample_index}"
                )
                self.assertLessEqual(max(reference_tiles_34), 4, diagnostic)
                self.assertEqual(lisjong_result, reference_result, diagnostic)


if __name__ == "__main__":
    unittest.main()
