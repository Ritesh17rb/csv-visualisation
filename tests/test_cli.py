from __future__ import annotations

import unittest

from csv_visualisation.cli import normalize_argv, split_option_values


class CliHelpersTest(unittest.TestCase):
    def test_split_option_values_flattens_and_deduplicates(self) -> None:
        self.assertEqual(
            split_option_values(["title,artist", "genre", "artist", "  year  "]),
            ["title", "artist", "genre", "year"],
        )

    def test_normalize_argv_strips_legacy_build_prefix(self) -> None:
        self.assertEqual(normalize_argv(["build", "music.csv", "--dry-run"]), ["music.csv", "--dry-run"])
        self.assertEqual(normalize_argv(["music.csv", "--dry-run"]), ["music.csv", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
