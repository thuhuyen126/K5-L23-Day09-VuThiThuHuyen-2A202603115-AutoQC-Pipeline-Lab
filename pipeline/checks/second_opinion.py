
"""CHECK 6 — Ý kiến thứ hai: so nhãn với model lớn hơn."""

from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    rules = cfg["second_opinion"]

    conf_threshold = rules["conf"]
    min_area_px = rules["min_area_px"]
    fp_conf = rules["fp_conf"]
    iou_missing = rules["iou_missing"]
    iou_match = rules["iou_match"]
    iou_loose = rules["iou_loose"]

    issues = []

    # Duyệt ảnh theo thứ tự ds.image_files()
    for image_path in ds.image_files():
        stem = image_path.stem

        # Bỏ qua ảnh không có dự đoán hoặc không có file nhãn
        if stem not in preds:
            continue

        label_lines = ds.read_label_lines(stem)
        if label_lines is None:
            continue

        image_size = ds.image_size(stem)
        if image_size is None:
            continue

        image_width, image_height = image_size

        # Đọc nhãn hợp lệ và giữ số dòng gốc
        labels = []

        for line_no, line in label_lines:
            try:
                box = parse_yolo_line(line)
            except ValueError:
                continue

            class_id, cx, cy, w, h = box

            if not 0 <= class_id < len(ds.classes):
                continue

            labels.append((line_no, class_id, (cx, cy, w, h)))

        # Chuẩn bị hai tập dự đoán:
        # P: dùng để phát hiện MISSING / LOOSE / WRONG_CLASS
        # fp_predictions: dùng để phát hiện FALSE_POS
        selected_preds = []
        fp_predictions = []

        for pred in preds[stem]:
            pred_class, pred_conf, cx, cy, w, h = pred
            box = (cx, cy, w, h)

            if pred_conf >= fp_conf:
                fp_predictions.append(box)

            area_px = (w * image_width) * (h * image_height)

            if (
                pred_conf >= conf_threshold
                and area_px >= min_area_px
            ):
                selected_preds.append(
                    (pred_class, pred_conf, box)
                )

        # 1. Kiểm tra từng dự đoán của model
        for pred_class, pred_conf, pred_box in selected_preds:

            candidates = []

            for line_no, label_class, label_box in labels:
                iou = iou_xywh(pred_box, label_box)

                if iou >= iou_missing:
                    candidates.append(
                        (label_class, iou)
                    )

            # Không có nhãn nào khớp
            if not candidates:
                issues.append(Issue(
                    "second_opinion",
                    WARN,
                    "POSSIBLE_MISSING",
                    stem,
                    "Model phát hiện đối tượng nhưng không tìm thấy nhãn tương ứng",
                    extra={"conf": pred_conf},
                ))
                continue

            # Tìm IoU tốt nhất cùng lớp và khác lớp
            same_class = [
                iou for label_class, iou in candidates
                if label_class == pred_class
            ]

            other_class = [
                iou for label_class, iou in candidates
                if label_class != pred_class
            ]

            # Có nhãn cùng lớp: ưu tiên kiểm tra nhãn cùng lớp
            if same_class:
                best_same_iou = max(same_class)

                if best_same_iou < iou_loose:
                    issues.append(Issue(
                        "second_opinion",
                        WARN,
                        "POSSIBLE_LOOSE_BOX",
                        stem,
                        "Box cùng lớp có IoU thấp với dự đoán của model",
                        extra={"iou": best_same_iou, "conf": pred_conf},
                    ))

                continue

            # Không có nhãn cùng lớp, kiểm tra nhãn khác lớp
            if other_class:
                best_other_iou = max(other_class)

                if best_other_iou >= iou_match:
                    issues.append(Issue(
                        "second_opinion",
                        WARN,
                        "POSSIBLE_WRONG_CLASS",
                        stem,
                        "Model và nhãn có khả năng gán khác lớp cho cùng đối tượng",
                        extra={"iou": best_other_iou, "conf": pred_conf},
                    ))

        # 2. Kiểm tra FALSE_POS cho từng dòng nhãn hợp lệ
        for line_no, label_class, label_box in labels:
            best_iou = max(
                (
                    iou_xywh(label_box, pred_box)
                    for pred_box in fp_predictions
                ),
                default=0.0,
            )

            if best_iou < iou_missing:
                issues.append(Issue(
                    "second_opinion",
                    WARN,
                    "POSSIBLE_FALSE_POS",
                    stem,
                    "Nhãn không khớp với dự đoán nào của model",
                    line=line_no,
                    extra={"iou": best_iou},
                ))

    return issues
