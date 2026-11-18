
"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.

Test: pytest tests/test_label_format.py -q
"""

from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line


def run(ds, cfg):
    issues = []
    rules = cfg["label_format"]

    overflow_tol = rules["overflow_tol"]
    min_box_px = rules["min_box_px"]
    max_box_frac = rules["max_box_frac"]
    dup_box_iou = rules["dup_box_iou"]

    for label_file in ds.label_files():
        stem = label_file.stem
        previous_boxes = []

        for line_no, line in ds.read_label_lines(stem) or []:
            # 1. Kiểm tra định dạng YOLO
            try:
                class_id, cx, cy, w, h = parse_yolo_line(line)
            except ValueError as exc:
                issues.append(Issue(
                    "label_format", ERROR, "BAD_FORMAT", stem,
                    f"Dòng nhãn sai định dạng: {exc}",
                    line=line_no
                ))
                continue

            # 2. Kiểm tra class ID
            if not 0 <= class_id < len(ds.classes):
                issues.append(Issue(
                    "label_format", ERROR, "BAD_CLASS_ID", stem,
                    f"Class ID {class_id} không hợp lệ",
                    line=line_no
                ))
                continue

            # 3. Kiểm tra box rỗng
            if w <= 0 or h <= 0:
                issues.append(Issue(
                    "label_format", ERROR, "BOX_ZERO", stem,
                    "Chiều rộng hoặc chiều cao box <= 0",
                    line=line_no
                ))
                continue

            # 4. Kiểm tra box vượt khỏi ảnh
            left = cx - w / 2
            right = cx + w / 2
            top = cy - h / 2
            bottom = cy + h / 2

            overflow = max(
                0.0,
                -left,
                -top,
                right - 1.0,
                bottom - 1.0
            )

            if overflow > overflow_tol:
                issues.append(Issue(
                    "label_format", ERROR, "BOX_OUT_OF_IMAGE", stem,
                    f"Box vượt khỏi ảnh: {overflow:.6f}",
                    line=line_no,
                    extra={"overflow": overflow}
                ))

            # 5. Kiểm tra box trùng với dòng trước đó
            current_box = (cx, cy, w, h)

            for prev_class, prev_box in previous_boxes:
                if (
                    class_id == prev_class
                    and iou_xywh(current_box, prev_box) > dup_box_iou
                ):
                    issues.append(Issue(
                        "label_format", ERROR, "DUP_BOX", stem,
                        "Box trùng với một dòng nhãn trước đó",
                        line=line_no
                    ))
                    break

            previous_boxes.append((class_id, current_box))

            # 6. Kiểm tra box quá nhỏ
            image_size = ds.image_size(stem)

            if image_size is not None:
                image_width, image_height = image_size

                box_width_px = w * image_width
                box_height_px = h * image_height

                if min(box_width_px, box_height_px) < min_box_px:
                    issues.append(Issue(
                        "label_format", WARN, "BOX_TINY", stem,
                        "Cạnh ngắn của box nhỏ hơn min_box_px",
                        line=line_no
                    ))

            # 7. Kiểm tra box quá lớn
            if w * h > max_box_frac:
                issues.append(Issue(
                    "label_format", WARN, "BOX_HUGE", stem,
                    "Diện tích box vượt max_box_frac",
                    line=line_no
                ))

    return issues
