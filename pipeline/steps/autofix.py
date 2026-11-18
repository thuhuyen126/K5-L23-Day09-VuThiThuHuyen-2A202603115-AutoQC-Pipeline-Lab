
"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN."""

from collections import defaultdict

import yaml

from ..common import ERROR, parse_yolo_line


def fix(ds, issues, cfg):
    fixes = []

    # Chỉ xử lý lỗi bắt buộc ERROR
    errors = [issue for issue in issues if issue.severity == ERROR]

    # Các lỗi cần xóa dòng nhãn
    drop_codes = {
        "BAD_FORMAT",
        "BAD_CLASS_ID",
        "BOX_ZERO",
        "DUP_BOX",
    }

    # Các lỗi cần xóa ảnh
    remove_codes = {
        "LEAK_TEST",
        "CORRUPT_IMAGE",
        "BAD_SIZE",
    }

    # Phân nhóm lỗi theo ảnh và dòng gốc
    drop_lines = defaultdict(dict)
    clip_lines = defaultdict(dict)
    remove_images = set()
    missing_labels = set()
    rename_classes = False

    for issue in errors:
        code = issue.code
        image = issue.image

        if code in drop_codes:
            if issue.line is not None:
                drop_lines[image][issue.line] = code

        elif code == "BOX_OUT_OF_IMAGE":
            if issue.line is not None:
                clip_lines[image][issue.line] = code

        elif code.startswith("DUP_IMAGE_") or code in remove_codes:
            remove_images.add(image)

        elif code == "MISSING_LABEL_FILE":
            missing_labels.add(image)

        elif code == "BAD_CLASS_NAME":
            rename_classes = True

    # 1. Xóa ảnh và file nhãn tương ứng
    for image in sorted(remove_images):
        image_path = ds.image_path(image)

        # Ảnh đã bị xóa thì bỏ qua
        if image_path is None:
            continue

        label_path = ds.label_path(image)

        before = image_path.name

        image_path.unlink()

        if label_path.exists():
            label_path.unlink()

        fixes.append({
            "image": image,
            "line": None,
            "action": "remove_image",
            "before": before,
            "after": "(đã xóa ảnh và nhãn)",
        })

    # 2. Xử lý các lỗi dòng nhãn
    affected_images = set(drop_lines) | set(clip_lines)

    for image in sorted(affected_images):

        # Ảnh đã bị xóa thì bỏ qua các issue dòng của nó
        if image in remove_images:
            continue

        label_path = ds.label_path(image)

        if not label_path.exists():
            continue

        original_lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        new_lines = []
        changed = False

        # Duyệt theo số dòng gốc, không đánh lại số dòng
        for line_no, original in enumerate(original_lines, start=1):

            # Dòng cần xóa luôn được ưu tiên hơn clip
            if line_no in drop_lines[image]:
                fixes.append({
                    "image": image,
                    "line": line_no,
                    "action": "drop_line",
                    "before": original,
                    "after": "",
                })
                changed = True
                continue

            # Dòng không cần clip thì giữ nguyên
            if line_no not in clip_lines[image]:
                new_lines.append(original)
                continue

            # Cắt box về trong [0, 1]
            try:
                class_id, cx, cy, w, h = parse_yolo_line(original)
            except ValueError:
                new_lines.append(original)
                continue

            x1 = max(0.0, cx - w / 2)
            y1 = max(0.0, cy - h / 2)
            x2 = min(1.0, cx + w / 2)
            y2 = min(1.0, cy + h / 2)

            # Box không còn diện tích sau khi clip
            if x2 <= x1 or y2 <= y1:
                fixes.append({
                    "image": image,
                    "line": line_no,
                    "action": "drop_line",
                    "before": original,
                    "after": "",
                })
                changed = True
                continue

            new_cx = (x1 + x2) / 2
            new_cy = (y1 + y2) / 2
            new_w = x2 - x1
            new_h = y2 - y1

            clipped = (
                f"{class_id} "
                f"{new_cx:.6f} "
                f"{new_cy:.6f} "
                f"{new_w:.6f} "
                f"{new_h:.6f}"
            )

            new_lines.append(clipped)

            fixes.append({
                "image": image,
                "line": line_no,
                "action": "clip_box",
                "before": original,
                "after": clipped,
            })

            changed = True

        # Chỉ ghi file một lần sau khi xử lý xong
        if changed:
            label_path.write_text(
                "\n".join(new_lines) + ("\n" if new_lines else ""),
                encoding="utf-8",
            )

    # 3. Sửa tên class trong data.yaml
    if rename_classes and ds.yaml_path.exists():
        data = yaml.safe_load(
            ds.yaml_path.read_text(encoding="utf-8")
        ) or {}

        before = data.get("names")

        data["names"] = {
            i: name
            for i, name in enumerate(cfg["classes"])
        }

        ds.yaml_path.write_text(
            yaml.safe_dump(
                data,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        fixes.append({
            "image": "data.yaml",
            "line": None,
            "action": "rename_classes",
            "before": str(before),
            "after": str(data["names"]),
        })

    # 4. Chuyển ảnh thiếu nhãn cho người gán
    for image in sorted(missing_labels):
        fixes.append({
            "image": image,
            "line": None,
            "action": "needs_label",
            "before": "(không có file)",
            "after": "chuyển người gán",
        })

    return fixes
