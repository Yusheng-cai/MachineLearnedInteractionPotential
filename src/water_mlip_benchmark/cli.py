from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from water_mlip_benchmark.config import load_config
from water_mlip_benchmark.convert import convert_archive_to_extxyz
from water_mlip_benchmark.data_sources import probe_zip_archive
from water_mlip_benchmark.mace import build_mace_train_command
from water_mlip_benchmark.split import split_extxyz


def _config_summary(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(f"project_root: {config.project_root}")
    print(f"raw_data_dir: {config.raw_data_dir}")
    print(f"processed_data_dir: {config.processed_data_dir}")
    print(f"runs_dir: {config.runs_dir}")
    print(f"figures_dir: {config.figures_dir}")
    print(f"random_seed: {config.random_seed}")
    return 0


def _probe_archive(args: argparse.Namespace) -> int:
    summary = probe_zip_archive(args.archive)
    print(f"archive_path: {summary.archive_path}")
    print(f"file_count: {summary.file_count}")
    print(f"total_uncompressed_bytes: {summary.total_uncompressed_bytes}")
    print("files:")
    for name in summary.names:
        print(f"  - {name}")
    return 0


def _convert(args: argparse.Namespace) -> int:
    count = convert_archive_to_extxyz(args.archive, args.output)
    print(f"converted_frames: {count}")
    print(f"output: {args.output}")
    return 0


def _mace_train_command(args: argparse.Namespace) -> int:
    command = build_mace_train_command(args.train_file, args.valid_file, args.config, args.run_dir)
    print(" ".join(command))
    return 0


def _split(args: argparse.Namespace) -> int:
    result = split_extxyz(
        args.source,
        args.output_dir,
        train_fraction=args.train_fraction,
        validation_fraction=args.valid_fraction,
        seed=args.seed,
    )
    print(f"train: {result.counts['train']} {result.train_file}")
    print(f"valid: {result.counts['valid']} {result.valid_file}")
    print(f"test: {result.counts['test']} {result.test_file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="water-mlip")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config-summary")
    config_parser.add_argument("config", type=Path)
    config_parser.set_defaults(func=_config_summary)

    probe_parser = subparsers.add_parser("probe-archive")
    probe_parser.add_argument("archive", type=Path)
    probe_parser.set_defaults(func=_probe_archive)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("archive", type=Path)
    convert_parser.add_argument("output", type=Path)
    convert_parser.set_defaults(func=_convert)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("source", type=Path)
    split_parser.add_argument("output_dir", type=Path)
    split_parser.add_argument("--train-fraction", type=float, default=0.8)
    split_parser.add_argument("--valid-fraction", type=float, default=0.1)
    split_parser.add_argument("--seed", type=int, default=20260518)
    split_parser.set_defaults(func=_split)

    mace_parser = subparsers.add_parser("mace-train-command")
    mace_parser.add_argument("train_file", type=Path)
    mace_parser.add_argument("valid_file", type=Path)
    mace_parser.add_argument("config", type=Path)
    mace_parser.add_argument("run_dir", type=Path)
    mace_parser.set_defaults(func=_mace_train_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
