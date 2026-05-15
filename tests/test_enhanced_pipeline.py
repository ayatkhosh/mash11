import unittest

from evaluate_enhanced_model import verify_city_split_isolation
from train_urbaninsight_v3_enhanced import (
    TrainConfig,
    split_city_tiles,
    stable_city_seed,
    verify_zero_data_leakage,
)


class TestEnhancedPipelineUtilities(unittest.TestCase):
    def test_stable_city_seed_is_deterministic_and_unique(self):
        seed_1 = stable_city_seed("Vegas")
        seed_2 = stable_city_seed("Vegas")
        seed_3 = stable_city_seed("Paris")
        self.assertEqual(seed_1, seed_2)
        self.assertNotEqual(seed_1, seed_3)

    def test_split_city_tiles_ratio_and_exclusivity(self):
        cfg = TrainConfig()
        tiles = [f"img_{i:04d}" for i in range(100)]
        splits = split_city_tiles(tiles, "Vegas", cfg)
        self.assertEqual(len(splits["train"]), 70)
        self.assertEqual(len(splits["val"]), 15)
        self.assertEqual(len(splits["test"]), 15)
        verify_zero_data_leakage({"Vegas": splits})

    def test_verify_city_split_isolation_flags_leakage(self):
        split_spec = {
            "Vegas": {
                "train": ["a", "b"],
                "val": ["c"],
                "test": ["b"],
            }
        }
        leakage = verify_city_split_isolation(split_spec)
        self.assertIn("Vegas", leakage)
        self.assertIn("train_test", leakage["Vegas"])


if __name__ == "__main__":
    unittest.main()
