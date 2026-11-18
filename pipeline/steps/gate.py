
"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework."""

from ..common import ERROR, WARN


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    # Tập hợp tên ảnh và ảnh đã được người xem kiểm tra
    image_stems = list(image_stems)
    image_set = set(image_stems)
    reviewed = set(reviewed)

    n_images = len(image_stems)

    # Đếm tất cả lỗi bắt buộc (ERROR).
    # Lỗi ERROR không được bỏ qua dù ảnh đã reviewed.
    n_errors = sum(
        1 for issue in issues
        if issue.severity == ERROR
    )

    # Đếm số ảnh khác nhau còn cảnh báo (WARN).
    # Chỉ tính ảnh thuộc dataset và chưa được reviewed.
    warn_images = {
        issue.image
        for issue in issues
        if (
            issue.severity == WARN
            and issue.image in image_set
            and issue.image not in reviewed
        )
    }

    n_warn_images = len(warn_images)

    # Tính tỷ lệ cảnh báo trên tổng số ảnh
    warn_ratio = (
        round(n_warn_images / n_images, 4)
        if n_images > 0
        else 0
    )

    warn_threshold = cfg["gate"]["warn_ratio_hold"]

    # Cây quyết định Release / Hold / Rework
    reasons = []

    if n_errors > 0:
        decision = "Rework"
        reasons.append(
            f"Có {n_errors} lỗi bắt buộc (ERROR), "
            f"cần sửa trước khi phát hành dataset."
        )

    elif warn_ratio >= warn_threshold:
        decision = "Hold"
        reasons.append(
            f"Có {n_warn_images}/{n_images} ảnh còn cảnh báo "
            f"(tỷ lệ {warn_ratio:.4f}), "
            f"đạt hoặc vượt ngưỡng Hold {warn_threshold:.4f}."
        )

    else:
        decision = "Release"
        reasons.append(
            f"Không có lỗi bắt buộc; "
            f"có {n_warn_images}/{n_images} ảnh còn cảnh báo "
            f"(tỷ lệ {warn_ratio:.4f}), "
            f"thấp hơn ngưỡng Hold {warn_threshold:.4f}."
        )

    return {
        "decision": decision,
        "n_errors": n_errors,
        "n_warn_images": n_warn_images,
        "n_images": n_images,
        "warn_ratio": warn_ratio,
        "reasons": reasons,
    }
