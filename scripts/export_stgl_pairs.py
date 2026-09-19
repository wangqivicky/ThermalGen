"""Export explicit paired RGB/thermal image patches from ThermalGen STGL."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image, ImageFile, ImageOps

# STGL stores very large map mosaics. Match the official dataset loader limit.
Image.MAX_IMAGE_PIXELS = 933_120_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_CONFIGS = {
    "Boson_night": "Boson_night.yml",
    "BosonPlus_day": "BosonPlus_day.yml",
    "BosonPlus_night": "BosonPlus_night.yml",
    "DJI_day": "DJI_day.yml",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export aligned RGB/thermal patch pairs from STGL maps."
    )
    parser.add_argument(
        "--stgl-root", type=Path,
        default=root / "datasets_preprocess" / "STGL",
        help="Directory containing folder_config.yml and maps/.",
    )
    parser.add_argument(
        "--config-root", type=Path, default=root / "configs" / "datasets",
        help="Directory containing the ThermalGen dataset YAML files.",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "datasets_paired" / "STGL",
        help="Output directory.",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASET_CONFIGS),
        default=list(DATASET_CONFIGS), help="STGL subsets to export.",
    )
    parser.add_argument(
        "--split", choices=("train", "val", "test", "all"), default="all",
        help="Dataset split to export.",
    )
    parser.add_argument(
        "--patch-size", type=int, default=512,
        help="Square crop size in source-map pixels (official loader: 512).",
    )
    parser.add_argument(
        "--stride", type=int, default=512,
        help="Distance between crop centers (official loader: 35).",
    )
    parser.add_argument(
        "--output-size", type=int, default=None,
        help="Optionally resize both patches to this square size, e.g. 256.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum pairs per map entry; useful for a quick inspection.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing image pairs instead of skipping them.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def map_spec(logical_name: str, folder_config: dict[str, Any]):
    try:
        data_name, index_text = logical_name.rsplit("_", 1)
        index = int(index_text)
        section = folder_config[data_name]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid map reference {logical_name!r}") from exc
    return index, section


def resolve_map_path(stgl_root: Path, section: dict[str, Any], index: int) -> Path:
    path = stgl_root / "maps" / str(section["name"]) / str(section["maps"][index])
    if path.is_file():
        return path

    # Some published validation entries are links to the corresponding train map.
    # ZIP/browser downloads on Windows can omit links, so resolve common aliases.
    candidates = []
    if "_val." in path.name:
        candidates.append(path.with_name(path.name.replace("_val.", "_train.")))
    if "_day_val." in path.name:
        candidates.append(path.with_name(path.name.replace("_day_val.", "_day_train.")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Map image not found: {path}\n"
        "Keep the Hugging Face maps/ directory unchanged. For linked validation "
        "files, download the repository with the Hugging Face CLI or Git LFS."
    )


def paired_centers(
    rgb_region: Iterable[int], thermal_region: Iterable[int],
    patch_size: int, stride: int,
) -> Iterable[tuple[int, int]]:
    rgb = list(map(int, rgb_region))
    thermal = list(map(int, thermal_region))
    top = max(rgb[0], thermal[0])
    left = max(rgb[1], thermal[1])
    bottom = min(rgb[2], thermal[2])
    right = min(rgb[3], thermal[3])
    half = patch_size // 2

    if patch_size <= 0 or patch_size % 2:
        raise ValueError("--patch-size must be a positive even integer")
    if stride <= 0:
        raise ValueError("--stride must be positive")
    if bottom - top < patch_size or right - left < patch_size:
        return

    # Same exclusive endpoint behavior as np.arange in STGLDataset.grid_sample.
    for y in range(top + half, bottom - half, stride):
        for x in range(left + half, right - half, stride):
            yield y, x


def save_pair(
    rgb_map: Image.Image, thermal_map: Image.Image, y: int, x: int,
    patch_size: int, output_size: int | None,
    rgb_path: Path, thermal_path: Path,
) -> None:
    half = patch_size // 2
    box = (x - half, y - half, x + half, y + half)
    rgb = rgb_map.crop(box)
    thermal = thermal_map.crop(box)
    if output_size is not None:
        if output_size <= 0:
            raise ValueError("--output-size must be positive")
        size = (output_size, output_size)
        rgb = rgb.resize(size, Image.Resampling.BILINEAR)
        thermal = thermal.resize(size, Image.Resampling.BILINEAR)
    rgb.save(rgb_path)
    thermal.save(thermal_path)


def export_entry(
    dataset_name: str, split: str, entry_index: int, entry: dict[str, Any],
    folder_config: dict[str, Any], args: argparse.Namespace,
) -> tuple[int, list[list[Any]]]:
    rgb_index, rgb_section = map_spec(entry["database_name"], folder_config)
    thermal_index, thermal_section = map_spec(entry["queries_name"], folder_config)
    rgb_path = resolve_map_path(args.stgl_root, rgb_section, rgb_index)
    thermal_path = resolve_map_path(args.stgl_root, thermal_section, thermal_index)
    rgb_region = rgb_section["valid_regions"][rgb_index]
    thermal_region = thermal_section["valid_regions"][thermal_index]

    base = args.output / dataset_name / split
    rgb_dir = base / "rgb"
    thermal_dir = base / "thermal"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    thermal_dir.mkdir(parents=True, exist_ok=True)

    rows: list[list[Any]] = []
    count = 0
    with Image.open(rgb_path) as rgb_source, Image.open(thermal_path) as thermal_source:
        rgb_map = ImageOps.exif_transpose(rgb_source).convert("RGB")
        thermal_map = ImageOps.exif_transpose(thermal_source).convert("L")
        for y, x in paired_centers(
            rgb_region, thermal_region, args.patch_size, args.stride
        ):
            if args.limit is not None and count >= args.limit:
                break
            stem = (
                f"{dataset_name}__{split}__m{entry_index:02d}"
                f"__y{y:06d}_x{x:06d}"
            )
            rgb_out = rgb_dir / f"{stem}.png"
            thermal_out = thermal_dir / f"{stem}.png"
            if args.overwrite or not (rgb_out.is_file() and thermal_out.is_file()):
                save_pair(
                    rgb_map, thermal_map, y, x, args.patch_size,
                    args.output_size, rgb_out, thermal_out,
                )
            rows.append([
                stem, dataset_name, split, entry["database_name"],
                entry["queries_name"], y, x,
                str(rgb_out.relative_to(args.output)),
                str(thermal_out.relative_to(args.output)),
            ])
            count += 1
    return count, rows


def main() -> int:
    args = parse_args()
    args.stgl_root = args.stgl_root.resolve()
    args.config_root = args.config_root.resolve()
    args.output = args.output.resolve()
    folder_config = load_yaml(args.stgl_root / "folder_config.yml")
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    manifest_rows: list[list[Any]] = []
    total = 0

    print(f"STGL source : {args.stgl_root}")
    print(f"Output      : {args.output}")
    print(f"Patch/stride: {args.patch_size}/{args.stride}")
    for dataset_name in args.datasets:
        dataset_cfg = load_yaml(args.config_root / DATASET_CONFIGS[dataset_name])
        for split in splits:
            entries = dataset_cfg.get(split) or []
            split_total = 0
            for entry_index, entry in enumerate(entries):
                count, rows = export_entry(
                    dataset_name, split, entry_index, entry, folder_config, args
                )
                split_total += count
                manifest_rows.extend(rows)
            print(f"{dataset_name:20s} {split:5s}: {split_total:7d} pairs")
            total += split_total

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sample_id", "dataset", "split", "rgb_map", "thermal_map",
            "center_y", "center_x", "rgb_path", "thermal_path",
        ])
        writer.writerows(manifest_rows)
    print(f"Done: {total} pairs")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)


