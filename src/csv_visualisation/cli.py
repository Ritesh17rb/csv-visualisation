from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .builder import DEFAULT_OUTPUT, DEFAULT_STATE_DB, DEFAULT_STATE_PARQUET, build_visualisation


def split_option_values(values: Sequence[str] | None) -> list[str]:
    flattened: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        for item in str(value).split(","):
            cleaned = item.strip()
            if not cleaned or cleaned in seen:
                continue
            flattened.append(cleaned)
            seen.add(cleaned)
    return flattened


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="csv-viz",
        description="Build a standalone HTML CSV visualisation.",
    )
    parser.add_argument("csv_input", nargs="?", help="Local CSV path or HTTP(S) URL")
    parser.add_argument("--csv", dest="csv_legacy", default="", help=argparse.SUPPRESS)
    parser.add_argument("--embedding-columns", action="append", default=[], help="Columns used to build embeddings")
    parser.add_argument("--image-columns", action="append", default=[], help="Image URL/path columns to describe and display")
    parser.add_argument("--audio-columns", action="append", default=[], help="Audio URL/path columns to describe and display")
    parser.add_argument("--audio-metadata-columns", action="append", default=[], help="Text columns to include when describing audio")
    parser.add_argument("--color-columns", action="append", default=[], help="Columns exposed for coloring")
    parser.add_argument("--filter-columns", action="append", default=[], help="Columns exposed for filtering")
    parser.add_argument("--cluster-columns", action="append", default=["embeddings"], help='Cluster dimensions, e.g. "embeddings,genre" or "genre"')
    parser.add_argument("--label-column", default="", help="Primary label column")
    parser.add_argument("--image-column", default="", help="Image URL column")
    parser.add_argument("--timeline-column", default="", help="Date or year column")
    parser.add_argument("--popup-style", choices=["auto", "grid", "list", "table"], default="auto", help="Default popup view")
    parser.add_argument("--clusters", type=int, default=0, help="Cluster count (0 = auto)")
    parser.add_argument("--name", "--branding", dest="name", default="", help="Dataset display name")
    parser.add_argument("--opacity", type=float, default=0.85, help="Base point opacity")
    parser.add_argument("--cluster-names", action="store_true", help="Ask Gemini to generate short cluster names")
    parser.add_argument("--cluster-naming-model", default="gemini-2.5-flash", help="Gemini model for cluster naming")
    parser.add_argument("--sample", type=int, default=0, help="Sample N rows before building")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Standalone HTML output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--state-db", default=str(DEFAULT_STATE_DB), help=f"DuckDB cache path (default: {DEFAULT_STATE_DB})")
    parser.add_argument(
        "--state-parquet",
        default=str(DEFAULT_STATE_PARQUET),
        help=f"Parquet export path (default: {DEFAULT_STATE_PARQUET})",
    )
    parser.add_argument("--no-export-parquet", action="store_true", help="Skip exporting the embedding cache to Parquet")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print a concise plan without embedding")
    return parser


def normalize_argv(argv: Sequence[str] | None) -> list[str]:
    args = list(argv if argv is not None else sys.argv[1:])
    if args and args[0] == "build":
        return args[1:]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(normalize_argv(argv))
    args.csv = args.csv_input or args.csv_legacy
    if not args.csv:
        parser.error("csv_input is required")
    args.embedding_columns = split_option_values(args.embedding_columns)
    args.image_columns = split_option_values(args.image_columns)
    args.audio_columns = split_option_values(args.audio_columns)
    args.audio_metadata_columns = split_option_values(args.audio_metadata_columns)
    args.color_columns = split_option_values(args.color_columns)
    args.filter_columns = split_option_values(args.filter_columns)
    args.cluster_columns = split_option_values(args.cluster_columns) or ["embeddings"]
    args.opacity = max(0.0, min(1.0, float(args.opacity)))
    build_visualisation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
