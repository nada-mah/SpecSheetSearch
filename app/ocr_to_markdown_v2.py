from __future__ import annotations

import logging
import re
from statistics import median
from typing import Any, Iterable

logger = logging.getLogger(__name__)

CLASS_NAMES = {
    0: "title", 1: "plain text", 2: "abandon",
    3: "figure", 4: "figure_caption", 5: "table",
    6: "table_caption", 7: "table_footnote",
    8: "isolate_formula", 9: "formula_caption",
}
SKIP_CLASSES = {"abandon"}


def _bbox(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)

def _center(poly):
    x1, y1, x2, y2 = _bbox(poly)
    return (x1 + x2) / 2, (y1 + y2) / 2

def _in_box(cx, cy, box):
    return box["x1"] <= cx <= box["x2"] and box["y1"] <= cy <= box["y2"]

def _token_from(txt, poly):
    x1, y1, x2, y2 = _bbox(poly)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return {"text": txt, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx, "cy": cy, "h": y2 - y1, "w": x2 - x1}


def _norm_layout(page_layout):
    if hasattr(page_layout, "summary"):
        return list(page_layout.summary())
    if isinstance(page_layout, list):
        return page_layout
    raise TypeError(f"Unsupported layout type: {type(page_layout)}")


def _norm_boxes(raw):
    out = []
    for b in raw:
        name = b.get("name") or CLASS_NAMES.get(b.get("class", -1), "unknown")
        if "box" not in b:
            continue
        out.append({"name": name, "class": b.get("class"),
                    "confidence": b.get("confidence", 0.0),
                    "box": {k: float(b["box"][k]) for k in ("x1","y1","x2","y2")}})
    return out


def _reading_order(boxes, row_tol=20.0):
    def key(b):
        y1 = b["box"]["y1"]
        return (round(y1 / row_tol) * row_tol, b["box"]["x1"])
    return sorted(boxes, key=key)


def _assign_tokens(texts, polys, boxes, page_h):
    abandon_zones = []
    non_abandon = []
    for b in boxes:
        if b["name"] in SKIP_CLASSES:
            abandon_zones.append(b["box"])
        else:
            non_abandon.append(b)

    assigned = {i: [] for i in range(len(boxes))}
    unassigned = []

    page_margin_top = page_h * 0.08
    page_margin_bot = page_h * 0.92

    for txt, poly in zip(texts, polys):
        if not txt or poly is None or len(poly) == 0:
            continue
        tok = _token_from(txt, poly)
        cx, cy = tok["cx"], tok["cy"]

        if any(_in_box(cx, cy, az) for az in abandon_zones):
            continue

        in_fringe = cy < page_margin_top or cy > page_margin_bot

        best_idx, best_area = -1, float("inf")
        for idx, b in enumerate(boxes):
            if b["name"] in SKIP_CLASSES:
                continue
            if _in_box(cx, cy, b["box"]):
                bx = b["box"]
                area = (bx["x2"] - bx["x1"]) * (bx["y2"] - bx["y1"])
                if area < best_area:
                    best_area = area; best_idx = idx

        if best_idx >= 0:
            assigned[best_idx].append(tok)
        elif not in_fringe:
            unassigned.append(tok)

    return assigned, unassigned


def _row_tol(tokens):
    hs = [t["h"] for t in tokens if t["h"] > 2]
    return (median(hs) * 0.65) if hs else 10.0


def _cluster_rows(tokens, tol=None):
    if not tokens:
        return []
    if tol is None:
        tol = _row_tol(tokens)
    srt = sorted(tokens, key=lambda t: t["cy"])
    rows = [[srt[0]]]
    for tok in srt[1:]:
        if abs(tok["cy"] - rows[-1][-1]["cy"]) <= tol:
            rows[-1].append(tok)
        else:
            rows.append([tok])
    for r in rows:
        r.sort(key=lambda t: t["x1"])
    return rows


def _row_text(row):
    return " ".join(t["text"] for t in row)


def _tokens_to_para(tokens):
    rows = _cluster_rows(tokens)
    return "\n".join(_row_text(r) for r in rows)


_SECTION_HEAD_RE = re.compile(r'^[A-Z][A-Z0-9 /\-&\'"\.]{2,40}$')

def _is_section_head(text):
    return bool(_SECTION_HEAD_RE.match(text.strip()))


def _detect_columns(tokens, page_w, gap_ratio=0.04):
    if not tokens:
        return []

    bin_w = max(1.0, page_w * gap_ratio)
    n_bins = int(page_w / bin_w) + 1
    hist = [0] * n_bins
    for t in tokens:
        b = int(t["x1"] / bin_w)
        e = int(t["x2"] / bin_w)
        for i in range(b, min(e + 1, n_bins)):
            hist[i] += 1

    in_gap = False
    gaps = []
    gap_start = 0
    for i, v in enumerate(hist):
        if v == 0 and not in_gap:
            in_gap = True; gap_start = i
        elif v > 0 and in_gap:
            in_gap = False
            gap_w = (i - gap_start) * bin_w
            if gap_w >= page_w * gap_ratio:
                gaps.append((gap_start * bin_w, i * bin_w))
    if in_gap:
        gap_w = (n_bins - gap_start) * bin_w
        if gap_w >= page_w * gap_ratio:
            gaps.append((gap_start * bin_w, page_w))

    if not gaps:
        return []

    bands = []
    prev = 0.0
    for g_start, g_end in gaps:
        if g_start > prev:
            bands.append((prev, g_start))
        prev = g_end
    if prev < page_w:
        bands.append((prev, page_w))

    return bands if len(bands) >= 2 else []


def _split_into_columns(tokens, bands):
    cols = [[] for _ in bands]
    for t in tokens:
        cx = t["cx"]
        best = min(range(len(bands)), key=lambda i: abs((bands[i][0] + bands[i][1]) / 2 - cx))
        cols[best].append(t)
    return cols


def _is_config_table(tokens, page_w, header_y_tol=60):
    rows = _cluster_rows(tokens, tol=20)
    if len(rows) < 2:
        return None, None

    for ri, row in enumerate(rows[:5]):
        head_texts = [t["text"].strip() for t in row]
        all_caps = sum(1 for s in head_texts if re.match(r'^[A-Z0-9/ \.]+$', s) and len(s) >= 2)
        if all_caps < 3:
            continue
        x_span = max(t["x2"] for t in row) - min(t["x1"] for t in row)
        if x_span < page_w * 0.5:
            continue
        body_tokens = [t for r2 in rows[ri + 1:] for t in r2]
        return row, body_tokens

    return None, None


def _build_config_table_md(header_row, body_tokens, page_w):
    if not header_row or not body_tokens:
        return ""

    col_centers = sorted(t["cx"] for t in header_row)
    col_labels = [t["text"].strip() for t in sorted(header_row, key=lambda t: t["cx"])]
    n = len(col_labels)

    def nearest_col(tok):
        return min(range(n), key=lambda i: abs(col_centers[i] - tok["cx"]))

    rows = _cluster_rows(body_tokens, tol=_row_tol(body_tokens) * 1.2 if body_tokens else 12)

    avg_tprow = len(body_tokens) / max(len(rows), 1)
    use_deflist = avg_tprow > n * 2.5

    lines = []
    lines.append("| " + " | ".join(col_labels) + " |")
    lines.append("| " + " | ".join(["---"] * n) + " |")
    for row in rows:
        cells = [""] * n
        for tok in row:
            ci = nearest_col(tok)
            if use_deflist:
                cells[ci] = (cells[ci] + " " + tok["text"]).strip() if cells[ci] else tok["text"]
            else:
                sep = " / " if cells[ci] else ""
                cells[ci] = cells[ci] + sep + tok["text"]
        cells = [c.replace("|", "\\|") for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_plain(tokens, page_w, region_box=None):
    if not tokens:
        return ""

    reg_w = page_w
    if region_box:
        reg_w = region_box["x2"] - region_box["x1"]

    bands = _detect_columns(tokens, reg_w, gap_ratio=0.05)

    if not bands:
        return _tokens_to_para(tokens)

    cols = _split_into_columns(tokens, bands)
    non_empty = [(i, col) for i, col in enumerate(cols) if col]

    if len(non_empty) == 1:
        return _tokens_to_para(non_empty[0][1])

    parts = []
    for _, col in non_empty:
        col_rows = _cluster_rows(col)
        if not col_rows:
            continue
        first_line = _row_text(col_rows[0])
        rest = "\n".join(_row_text(r) for r in col_rows[1:])
        if _is_section_head(first_line):
            block = f"### {first_line}\n{rest}".rstrip()
        else:
            block = "\n".join(_row_text(r) for r in col_rows)
        parts.append(block)

    return "\n\n".join(parts)


def _escape(text):
    return text.replace("|", "\\|")


def _render_plain_table(tokens, page_w):
    rows = _cluster_rows(tokens)
    if not rows:
        return ""

    widths = [t["w"] for row in rows for t in row if t["w"] > 2]
    tol = (median(widths) * 0.4) if widths else 12.0
    lefts = sorted(t["x1"] for row in rows for t in row)
    clusters = [[lefts[0]]]
    for v in lefts[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    min_sup = max(1, int(len(rows) * 0.2))
    col_centers = [sum(float(v) for v in c) / len(c) for c in clusters if len(c) >= min_sup]
    if not col_centers:
        col_centers = [sum(float(v) for v in clusters[0]) / len(clusters[0])]
    n = len(col_centers)

    def nc(tok):
        return min(range(n), key=lambda i: abs(col_centers[i] - tok["x1"]))

    grid = []
    for row in rows:
        cells = [""] * n
        for tok in row:
            ci = nc(tok)
            cells[ci] = (cells[ci] + " " + tok["text"]).strip() if cells[ci] else tok["text"]
        grid.append([_escape(c) for c in cells])

    lines = ["| " + " | ".join(grid[0]) + " |",
             "| " + " | ".join(["---"] * n) + " |"]
    for r in grid[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _render_region(box, tokens, page_w, page_h):
    name = box["name"]
    if not tokens:
        return "" if name != "figure" else "*[figure]*"

    if name == "title":
        rows = _cluster_rows(tokens)
        text = " ".join(_row_text(r) for r in rows)
        return f"## {text}".strip()

    if name == "table":
        header_row, body_tokens = _is_config_table(tokens, page_w)
        if header_row is not None:
            return _build_config_table_md(header_row, body_tokens, page_w)
        return _render_plain_table(tokens, page_w)

    if name == "table_caption":
        rows = _cluster_rows(tokens)
        return "**" + " ".join(_row_text(r) for r in (rows or [[]])) + "**"

    if name in ("table_footnote", "figure_caption", "formula_caption"):
        rows = _cluster_rows(tokens)
        return "*" + " ".join(_row_text(r) for r in rows) + "*"

    if name == "figure":
        rows = _cluster_rows(tokens)
        inner = " ".join(_row_text(r) for r in rows)
        return f"*[figure: {inner}]*" if inner else "*[figure]*"

    if name == "isolate_formula":
        return "```\n" + _tokens_to_para(tokens) + "\n```"

    if name in ("plain text", "unknown"):
        reg_box = box.get("box")
        return _render_plain(tokens, page_w, region_box=reg_box)

    return _tokens_to_para(tokens)


def _render_stray(tokens, page_w):
    if not tokens:
        return ""
    return _render_plain(tokens, page_w)


def ocr_page_to_markdown(
    ocr_page: dict,
    layout_page: Any,
    *,
    page_index: int = 0,
    page_width: float = 2550.0,
    page_height: float = 3300.0,
) -> str:
    """
    Convert a single OCR page dict + doclayout_yolo result into Markdown.

    ocr_page must have 'rec_texts' and 'rec_polys' keys (same structure as
    get_ocr_object_per_page_paddle output: ocr_results[i][0]).
    """
    pw = float(ocr_page.get("image_width", page_width))
    ph = float(ocr_page.get("image_height", page_height))

    boxes_raw = _norm_layout(layout_page)
    boxes = _norm_boxes(boxes_raw)
    sorted_boxes = _reading_order(boxes)

    assigned, unassigned = _assign_tokens(
        ocr_page.get("rec_texts", []),
        ocr_page.get("rec_polys", []),
        boxes, ph
    )

    chunks: list[str] = [f"<!-- Page {page_index + 1} -->"]

    for box in sorted_boxes:
        if box["name"] in SKIP_CLASSES:
            continue
        orig_idx = boxes.index(box)
        toks = assigned.get(orig_idx, [])
        chunk = _render_region(box, toks, pw, ph)
        if chunk:
            chunks.append(chunk)

    if unassigned:
        stray = _render_stray(unassigned, pw)
        if stray:
            chunks.append(stray)

    return "\n\n".join(chunks)


def ocr_to_markdown(
    ocr_pages: list[dict],
    layout_results: Iterable[Any],
    *,
    document_title: str | None = None,
    include_page_breaks: bool = True,
    page_width: float = 2550.0,
    page_height: float = 3300.0,
) -> str:
    layout_list = list(layout_results)
    if len(layout_list) != len(ocr_pages):
        raise ValueError(
            f"Page count mismatch: OCR={len(ocr_pages)}, layout={len(layout_list)}"
        )

    md: list[str] = []
    if document_title:
        md.append(f"# {document_title}\n")

    for page_idx, (ocr_page, layout_page) in enumerate(zip(ocr_pages, layout_list)):
        page_md = ocr_page_to_markdown(
            ocr_page, layout_page,
            page_index=page_idx,
            page_width=page_width,
            page_height=page_height,
        )
        if include_page_breaks and page_idx > 0:
            md.append("---\n\n" + page_md)
        else:
            md.append(page_md)

    return "\n\n".join(md).rstrip() + "\n"
