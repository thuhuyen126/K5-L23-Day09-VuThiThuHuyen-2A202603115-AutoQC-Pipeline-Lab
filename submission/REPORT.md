# Báo cáo Day 09 — Vũ Thị Thu Huyền, 2A202603115

## 1. Kết luận trước

Pipeline kết luận **HOLD**. Intake quét 224 ảnh, giữ lại 193 ảnh và loại 31 ảnh. Sau QC, autofix và review mô phỏng, còn 34/193 ảnh có cảnh báo (17.62%), vượt ngưỡng Gate 10%. Vì vậy, `clean_v1` chỉ là dataset xuất thử nghiệm, chưa đủ điều kiện Release.

## 2. Lỗi và phân việc

Thống kê từ `02_error_list_v1.txt` trước sửa: 67 dòng ERROR và 149 dòng WARN trên 193 ảnh được quét. Đây là số dòng lỗi/cảnh báo, không phải số ảnh duy nhất.

| Mã lỗi | Số dòng | Xử lý | Lý do |
|---|---:|---|---|
| BAD_CLASS_ID | 6 | Máy | Class ID không hợp lệ |
| BAD_CLASS_NAME | 2 | Máy | Tên lớp sai |
| BAD_FORMAT | 3 | Máy | Dòng nhãn sai định dạng |
| BAD_SIZE | 3 | Máy | Ảnh không đạt kích thước |
| BOX_OUT_OF_IMAGE | 8 | Máy | Box vượt biên ảnh |
| BOX_ZERO | 4 | Máy | Box có kích thước bằng 0 |
| CORRUPT_IMAGE | 2 | Máy | Ảnh không đọc được |
| DUP_BOX | 7 | Máy | Box trùng |
| DUP_IMAGE_EXACT | 8 | Máy | Ảnh trùng chính xác |
| DUP_IMAGE_NEAR | 7 | Máy | Ảnh trùng gần đúng |
| LEAK_TEST | 11 | Máy | Ảnh trùng tập test/val |
| MISSING_LABEL_FILE | 6 | Người | Cần xác minh và bổ sung nhãn |

Hàng đợi có 121 ảnh, ngân sách review 80 ảnh. `img_0020` đứng đầu (priority 11.524) vì đồng thời có `FIXED_DROP_LINE`, `POSSIBLE_MISSING`, `POSSIBLE_WRONG_CLASS`. Các ảnh `NEEDS_LABEL` tiếp theo có priority 10.000. Review trong lần chạy này là mô phỏng, không phải 80 ảnh đã được con người xác nhận và sửa lỗi ngữ nghĩa.

## 3. Calibration

| Tham số | Ban đầu | Đã chọn | Precision / Recall trước → sau | Lý do |
|---|---:|---:|---|---|
| `duplicates.phash_dist` | 2 | 16 | 1.00/0.50 → 1.00/0.83 (near duplicate) | Tăng Recall, giữ Precision; F1 tăng 0.67 → 0.91 |
| `second_opinion.conf` | 0.25 | 0.60 | 0.56/0.83 → 0.62/0.83 (wrong class) | Tăng Precision, giữ Recall cho nhóm này |

Ngưỡng 0.60 không cải thiện tất cả nhóm: `POSSIBLE_MISSING` đạt P/R = 0.57/1.00, nhưng `POSSIBLE_LOOSE_BOX` chỉ đạt 0.20/0.20. Đây là sự đánh đổi giữa các loại lỗi.

Điểm dễ đánh lừa: pHash đo độ tương đồng hình ảnh, không hiểu ngữ nghĩa. `img_0011` và `img_0008` có pHash distance 16 nhưng nội dung thực tế khác nhau (người chơi tennis và cặp người trượt tuyết). Do đó, cảnh báo trùng gần đúng cần được kiểm tra bằng mắt.

## 4. Gate và điểm mù

Chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb`: pipeline còn 6 lỗi bắt buộc, kết luận **REWORK**, mã thoát 20 và không xuất dataset. Điều này chứng minh Gate chặn xuất khi lỗi bắt buộc chưa được xử lý.

Hai điểm mù:
- `img_0011` và `img_0008` bị xem là trùng gần đúng dù nội dung khác nhau, cho thấy pHash có thể tạo false positive.
- `img_0078` có nhãn với tâm box x = 1.000018, vượt biên ảnh; IoU với prediction được đối chiếu khoảng 0.089. Đây là dấu hiệu bất thường cần người kiểm tra, không chứng minh prediction luôn đúng.
- Các cảnh báo `POSSIBLE_MISSING`, `POSSIBLE_FALSE_POS`, `POSSIBLE_WRONG_CLASS` phụ thuộc model; model cũng có thể dự đoán sai nên không thể tự động xem cảnh báo là lỗi nhãn đã xác nhận.

## 5. Train: Raw vs Clean

YOLO26n, 20 epochs, cùng cấu hình, hai seed (0 và 1) mỗi biến thể; đánh giá trên 100 ảnh test.

| Biến thể | Model | Epoch | Seeds | Ảnh train |
|---|---|---:|---|---:|
| Raw | YOLO26n | 20 | 0, 1 | 224 |
| Clean | YOLO26n | 20 | 0, 1 | 193 |

| Chỉ số | Raw (TB ± SD) | Clean (TB ± SD) |
|---|---:|---:|
| mAP50 | 0.726 ± 0.000 | 0.730 ± 0.001 |
| mAP50-95 | 0.512 ± 0.004 | 0.501 ± 0.001 |
| Precision | 0.764 ± 0.061 | 0.800 ± 0.019 |
| Recall | 0.649 ± 0.028 | 0.624 ± 0.012 |

Chênh mAP50 (Clean − Raw) = **+0.00435**. Mức tăng quan sát được lớn hơn SD riêng của từng biến thể, nhưng chỉ có hai seed nên chưa đủ cơ sở khái quát. Đồng thời mAP50-95 giảm 0.011 và Recall giảm 0.0247; vì vậy không thể kết luận Clean tốt hơn Raw trên mọi khía cạnh.

Các giả thuyết cần kiểm tra: (1) Raw có 11 trường hợp `LEAK_TEST`, ảnh trùng/rò rỉ có thể ảnh hưởng tính độc lập dữ liệu; cần rà soát các cặp train/test. (2) Clean vẫn có thể còn nhãn thiếu, box lỏng hoặc nhầm lớp; cần kiểm tra thủ công các cảnh báo còn lại. (3) Test chỉ có 100 ảnh; cần đánh giá trên tập lớn hơn để xác nhận xu hướng.

## 6. Quyết định và ghi chú AI

**Quyết định: HOLD.** Để tiến tới RELEASE, cần kiểm tra và xử lý các cảnh báo còn lại, đặc biệt là thiếu nhãn và lỗi ngữ nghĩa; thực hiện human review thực tế, chạy lại QC và Gate. Chỉ Release khi tỷ lệ cảnh báo không vượt 10% và không còn lỗi bắt buộc.

**Sử dụng AI:**
1. Prompt chính: Hỗ trợ triển khai và rà soát bốn module `label_format`, `second_opinion`, `autofix`, `gate` theo yêu cầu lab.
2. Kiểm chứng: Đọc code, chạy unit test từng module và toàn bộ pytest; đối chiếu calibration, báo cáo lỗi và kết quả pipeline.
3. AI từng diễn giải nhầm overlay annotation là prediction của model. Tôi kiểm tra `viz_labels.py` và xác nhận công cụ chỉ vẽ annotation, nên không dùng overlay đó làm bằng chứng prediction.
