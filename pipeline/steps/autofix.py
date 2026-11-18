"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401

REMOVE_CODES = {"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
DROP_LINE_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
LINE_CODES = DROP_LINE_CODES | {"BOX_OUT_OF_IMAGE"}


def fix(ds, issues, cfg):
    err_issues = [i for i in issues if i.severity == ERROR]
    fixes = []

    # 1. Sửa tên lớp trong data.yaml
    if any(i.code == "BAD_CLASS_NAME" for i in err_issues):
        cur_names = ds.yaml_names()
        expected = dict(enumerate(cfg["classes"]))
        if cur_names != expected:
            data = {}
            if ds.yaml_path.exists():
                data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            before_str = str(data.get("names", cur_names))
            data["names"] = expected
            ds.yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
            fixes.append({
                "image": "data.yaml",
                "line": "",
                "action": "rename_classes",
                "before": before_str,
                "after": str(cfg["classes"]),
            })

    # 2. Xoá ảnh lỗi / trùng / rò rỉ
    removed_stems = set()
    for i in err_issues:
        if i.code in REMOVE_CODES or i.code.startswith("DUP_IMAGE_"):
            stem = i.image
            if stem in removed_stems:
                continue
            removed_stems.add(stem)
            img_p = ds.image_path(stem)
            lbl_p = ds.label_path(stem)
            existed = False
            before_name = stem
            if img_p is not None and img_p.exists():
                before_name = img_p.name
                img_p.unlink()
                existed = True
            if lbl_p.exists():
                lbl_p.unlink()
                existed = True
            if existed:
                fixes.append({
                    "image": stem,
                    "line": "",
                    "action": "remove_image",
                    "before": before_name,
                    "after": "(đã xoá)",
                })

    # 3. Ghi nhận ảnh thiếu file nhãn (chuyển người gán, không tạo file rỗng)
    needs_seen = set()
    for i in err_issues:
        if i.code == "MISSING_LABEL_FILE" and i.image not in removed_stems and i.image not in needs_seen:
            needs_seen.add(i.image)
            fixes.append({
                "image": i.image,
                "line": "",
                "action": "needs_label",
                "before": "(không có file)",
                "after": "chuyển người gán",
            })

    # 4. Gom lỗi mức dòng theo từng file nhãn và sửa một lần trên số dòng gốc
    by_image = defaultdict(lambda: defaultdict(set))
    for i in err_issues:
        if i.code in LINE_CODES and i.line is not None and i.image not in removed_stems:
            by_image[i.image][i.line].add(i.code)

    for stem, line_map in by_image.items():
        lbl_p = ds.label_path(stem)
        if not lbl_p.exists():
            continue
        orig_lines = lbl_p.read_text(encoding="utf-8", errors="replace").splitlines()
        new_lines = []
        changed = False

        for idx, text in enumerate(orig_lines, 1):
            codes = line_map.get(idx)
            if not codes:
                new_lines.append(text)
                continue

            if codes & DROP_LINE_CODES:
                changed = True
                fixes.append({
                    "image": stem,
                    "line": idx,
                    "action": "drop_line",
                    "before": text,
                    "after": "",
                })
                continue

            if "BOX_OUT_OF_IMAGE" in codes:
                try:
                    cls_id, cx, cy, w, h = parse_yolo_line(text)
                except ValueError:
                    changed = True
                    fixes.append({
                        "image": stem,
                        "line": idx,
                        "action": "drop_line",
                        "before": text,
                        "after": "",
                    })
                    continue

                x1 = max(0.0, cx - w / 2)
                y1 = max(0.0, cy - h / 2)
                x2 = min(1.0, cx + w / 2)
                y2 = min(1.0, cy + h / 2)
                nw, nh = x2 - x1, y2 - y1

                if nw <= 0 or nh <= 0:
                    changed = True
                    fixes.append({
                        "image": stem,
                        "line": idx,
                        "action": "drop_line",
                        "before": text,
                        "after": "",
                    })
                else:
                    ncx, ncy = (x1 + x2) / 2, (y1 + y2) / 2
                    after_text = f"{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"
                    new_lines.append(after_text)
                    changed = True
                    fixes.append({
                        "image": stem,
                        "line": idx,
                        "action": "clip_box",
                        "before": text,
                        "after": after_text,
                    })

        if changed:
            content = "\n".join(new_lines) + ("\n" if new_lines else "")
            lbl_p.write_text(content, encoding="utf-8")

    return fixes
