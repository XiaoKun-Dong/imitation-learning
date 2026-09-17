"""Compose a paper-ready two-camera mosaic from newly rendered DemoVLA panels.

This utility never alters attention or RGB pixels. It crops the requested
query's base and left-wrist panels from renderer outputs, then places them on a
white canvas with figure-level task, camera, and replan labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PANEL_SIZE = 224
LABEL_HEIGHT = 24
DEFAULT_REPLANS = (0, 10, 20, 30, 40, 50, 60, 70, 77)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{filename}", size)


def _panel_path(source_dir: Path, replan: int) -> Path:
    matches = sorted(source_dir.glob(f"interaction_replan_{replan:03d}_step_*.png"))
    if len(matches) != 1:
        raise ValueError(f"expected one rendered image for replan {replan}; found {len(matches)}")
    return matches[0]


def _query_camera_panels(source_dir: Path, replan: int, query_index: int, num_queries: int) -> tuple[Image.Image, Image.Image]:
    with Image.open(_panel_path(source_dir, replan)) as rendered:
        image = rendered.convert("RGB")
    expected_height = num_queries * (PANEL_SIZE + LABEL_HEIGHT)
    title_height = image.height - expected_height
    if title_height < 0 or image.width < 2 * PANEL_SIZE:
        raise ValueError(f"unexpected renderer geometry {image.size} for replan {replan}")
    top = title_height + query_index * (PANEL_SIZE + LABEL_HEIGHT) + LABEL_HEIGHT
    return (
        image.crop((0, top, PANEL_SIZE, top + PANEL_SIZE)),
        image.crop((PANEL_SIZE, top, 2 * PANEL_SIZE, top + PANEL_SIZE)),
    )


def compose(
    source_dir: Path,
    output_path: Path,
    task: str,
    replans: tuple[int, ...],
    query_index: int,
    num_queries: int,
) -> None:
    # Reserve enough space for the two-line camera names at paper scale.
    left_margin, right_margin, gap = 150, 18, 8
    title_height, replan_height, row_gap, bottom_margin = 42, 34, 10, 16
    width = left_margin + len(replans) * PANEL_SIZE + (len(replans) - 1) * gap + right_margin
    height = title_height + replan_height + 2 * PANEL_SIZE + row_gap + bottom_margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = _font(23, bold=True)
    title = f"Task: {task}"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 8), title, fill="black", font=title_font)

    replan_font = _font(16, bold=True)
    row_font = _font(18, bold=True)
    row_tops = (title_height + replan_height, title_height + replan_height + PANEL_SIZE + row_gap)
    for column, replan in enumerate(replans):
        x = left_margin + column * (PANEL_SIZE + gap)
        label = f"replan {replan:03d}"
        label_box = draw.textbbox((0, 0), label, font=replan_font)
        draw.text((x + (PANEL_SIZE - (label_box[2] - label_box[0])) / 2, title_height + 5), label, fill="black", font=replan_font)
        for top, panel in zip(row_tops, _query_camera_panels(source_dir, replan, query_index, num_queries), strict=True):
            canvas.paste(panel, (x, top))

    for label, top in zip(("Base camera", "Left-wrist\ncamera"), row_tops, strict=True):
        label_box = draw.multiline_textbbox((0, 0), label, font=row_font, align="center", spacing=3)
        label_width, label_height = label_box[2] - label_box[0], label_box[3] - label_box[1]
        draw.multiline_text(
            (left_margin - label_width - 12, top + (PANEL_SIZE - label_height) / 2),
            label,
            fill="black",
            font=row_font,
            align="center",
            spacing=3,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--replans", type=int, nargs="+", default=DEFAULT_REPLANS)
    parser.add_argument("--query-index", type=int, default=3)
    parser.add_argument("--num-queries", type=int, default=4)
    args = parser.parse_args()
    compose(
        args.source_dir,
        args.output_path,
        args.task,
        tuple(args.replans),
        args.query_index,
        args.num_queries,
    )


if __name__ == "__main__":
    main()
