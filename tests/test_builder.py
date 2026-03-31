from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from csv_visualisation.builder import MediaRef, PreparedRow, build_vis_payload, cluster_with_columns, load_env


def row(label: str, genre: str, *, image: str = "", audio: str = "") -> PreparedRow:
    return PreparedRow(
        row_index=0,
        raw={"title": label, "genre": genre, "year": "2024"},
        label=label,
        text_payload=f"title: {label}. genre: {genre}",
        audio_metadata_text="",
        images=[MediaRef(kind="image", column="cover", raw_value=image, display_url=image)] if image else [],
        audios=[MediaRef(kind="audio", column="clip", raw_value=audio, display_url=audio)] if audio else [],
    )


class BuilderHelpersTest(unittest.TestCase):
    def test_load_env_accepts_bare_key(self) -> None:
        previous = os.environ.pop("GEMINI_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                env_path = os.path.join(tmpdir, ".env")
                with open(env_path, "w", encoding="utf-8") as handle:
                    handle.write("AIzaSyExampleBareKey\n")
                load_env(Path(env_path))
                self.assertEqual(os.environ.get("GEMINI_API_KEY"), "AIzaSyExampleBareKey")
        finally:
            os.environ.pop("GEMINI_API_KEY", None)
            if previous is not None:
                os.environ["GEMINI_API_KEY"] = previous

    def test_cluster_with_columns_supports_direct_labels(self) -> None:
        rows = [row("Song A", "rock"), row("Song B", "jazz"), row("Song C", "rock")]
        vectors = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.1]], dtype=np.float32)

        cluster_ids, label_map = cluster_with_columns(vectors, rows, ["genre"], n_clusters=0)

        self.assertEqual(cluster_ids.tolist(), [0, 1, 0])
        self.assertEqual(label_map, {0: "rock", 1: "jazz"})

    def test_build_vis_payload_uses_named_clusters_and_media_flags(self) -> None:
        rows = [
            row("Song A", "rock", image="https://example.com/a.jpg", audio="https://example.com/a.mp3"),
            row("Song B", "jazz"),
        ]
        points = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        cluster_ids = np.array([0, 1], dtype=int)
        cluster_labels = {0: "Rock Stories", 1: "Jazz Stories"}

        payload = build_vis_payload(
            rows,
            points,
            cluster_ids,
            cluster_labels,
            color_cols=["cluster", "genre"],
            filter_cols=["genre"],
            timeline_col="year",
            name="Songs",
            opacity=0.7,
            popup_style="list",
        )

        self.assertEqual(payload["points"][0]["cluster"], "Rock Stories")
        self.assertEqual(payload["points"][0]["clusterId"], 0)
        self.assertEqual(payload["points"][0]["audioCount"], 1)
        self.assertTrue(payload["meta"]["hasImages"])
        self.assertTrue(payload["meta"]["hasAudio"])
        self.assertEqual(payload["meta"]["opacity"], 0.7)
        self.assertEqual(payload["meta"]["popupStyle"], "list")


if __name__ == "__main__":
    unittest.main()
