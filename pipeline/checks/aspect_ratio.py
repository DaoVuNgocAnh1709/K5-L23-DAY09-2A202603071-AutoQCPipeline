"""Check mở rộng (Thưởng +3) — Kiểm tra lệch tỉ lệ khung hình box (BOX_EXTREME_ASPECT) và mất cân bằng lớp (CLASS_IMBALANCE).

Test: pytest tests/test_aspect_ratio.py -q

- `BOX_EXTREME_ASPECT` (WARN): tỉ lệ cạnh dài / cạnh ngắn theo pixel `max(w_px / h_px, h_px / w_px) > max_aspect_ratio` (mặc định 6.0).
  Bắt các box bị kéo dẹt bất thường (như sợi chỉ ngang/dọc) mà kiểm tra diện tích hoặc cạnh ngắn đơn thuần dễ bỏ sót.
- `CLASS_IMBALANCE` (WARN): tỉ số giữa lớp nhiều box nhất và lớp ít box nhất (trong các lớp hợp lệ có mặt) `> max_imbalance_ratio` (mặc định 5.0).
"""
from collections import Counter

from ..common import WARN, Issue, parse_yolo_line


def run(ds, cfg):
    ar_cfg = cfg.get("aspect_ratio", {})
    max_ar = ar_cfg.get("max_aspect_ratio", 6.0)
    max_imb = ar_cfg.get("max_imbalance_ratio", 5.0)
    n_cls = len(ds.classes)

    issues = []
    cls_counts = Counter()

    for lf in ds.label_files():
        stem = lf.stem
        size = ds.image_size(stem)
        W, H = size if size is not None else (1, 1)

        for line_no, text in ds.read_label_lines(stem) or []:
            try:
                cls_id, cx, cy, w, h = parse_yolo_line(text)
            except ValueError:
                continue
            if not (0 <= cls_id < n_cls) or w <= 0 or h <= 0:
                continue

            cls_counts[cls_id] += 1
            w_px, h_px = w * W, h * H
            ratio = max(w_px / h_px, h_px / w_px)
            if ratio > max_ar:
                issues.append(
                    Issue(
                        "aspect_ratio",
                        WARN,
                        "BOX_EXTREME_ASPECT",
                        stem,
                        f"tỉ lệ khung hình lệch {ratio:.2f} > {max_ar} ({w_px:.1f}x{h_px:.1f}px)",
                        line=line_no,
                        extra={"aspect_ratio": round(ratio, 3)},
                    )
                )

    if len(cls_counts) >= 2:
        hi_cls, hi_n = max(cls_counts.items(), key=lambda x: x[1])
        lo_cls, lo_n = min(cls_counts.items(), key=lambda x: x[1])
        if lo_n > 0 and (hi_n / lo_n) > max_imb:
            imb = round(hi_n / lo_n, 2)
            issues.append(
                Issue(
                    "aspect_ratio",
                    WARN,
                    "CLASS_IMBALANCE",
                    "<dataset>",
                    f"lệch phân phối lớp: {ds.classes[hi_cls]} ({hi_n} box) gấp {imb}x {ds.classes[lo_cls]} ({lo_n} box)",
                    extra={"max_class": ds.classes[hi_cls], "max_boxes": hi_n, "min_class": ds.classes[lo_cls], "min_boxes": lo_n, "ratio": imb},
                )
            )

    return issues
