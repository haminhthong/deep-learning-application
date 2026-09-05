# Dataset Card: Construction PPE Detection Dataset

## 1. Dữ Liệu Tổng Quan (Dataset Summary)
- **Nguồn dữ liệu (Source)**: Dữ liệu hình ảnh và video giám sát thu thập từ camera an ninh công trường xây dựng công nghiệp thực tế và dữ liệu benchmark PPE cộng đồng.
- **Bản quyền / License**: CC BY 4.0 / Educational & Research Portfolio Use.
- **Tổng số lượng**:
  - Số hình ảnh / Person Crops: **5,200 mẫu**.
  - Số phiên ghi hình: **14 recording sessions** độc lập.
- **Độ phân giải nguồn**: $1920 \times 1080$ (Full HD) và $1280 \times 720$ (HD 720p).
- **Phân bố bối cảnh**: Ban ngày tự nhiên (70%), Ánh sáng nhân tạo / nhà xưởng (20%), Thiếu sáng / hoàng hôn (10%).

---

## 2. Thống Kê Phân Phối Lớp (Class Statistics)

Dữ liệu được kiểm toán tự động bởi [`audit_dataset.py`](./audit_dataset.py) và kết xuất vào [`dataset_report.json`](./dataset_report.json):

| Class ID | Class Name | Train Set | Validation Set | Test Set | Total Instances |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | `helmet` | 3,240 | 690 | 710 | 4,640 |
| 1 | `no-helmet` | 400 | 90 | 70 | 560 |
| 2 | `vest` | 2,878 | 620 | 622 | 4,120 |
| 3 | `no-vest` | 762 | 160 | 158 | 1,080 |

> [!NOTE]
> Thống kê trên phản ánh chính xác số lượng nhãn bounding box sau kiểm toán. Mỗi mẫu Person Crop có thể chứa đồng thời cả nhãn vùng đầu (helmet / no-helmet) và nhãn vùng thân (vest / no-vest).

---

## 3. Định Nghĩa & Quy Tắc Gán Nhãn (Annotation Semantics)

Để tránh mơ hồ nhãn khi huấn luyện mô hình Stage-2 trên vùng ảnh cắt (Person ROI), taxonomy nhãn được chuẩn hóa nghiêm ngặt theo phân vùng cơ thể (Body-Zone Semantics):

- **`helmet`**: Bounding box bao quanh chính xác chiếc mũ bảo hộ lao động (hardhat/helmet) nằm trên vùng đầu (`y <= 35%` chiều cao người). Mũ bảo hiểm xe máy hoặc mũ vải thông thường không được coi là mũ bảo hộ.
- **`no-helmet`**: Bounding box bao quanh vùng đầu / tóc trần không đội mũ bảo hộ (`y <= 35%` chiều cao người).
- **`vest`**: Bounding box bao quanh áo bảo hộ phản quang (high-visibility safety vest) nằm trên thân người (`30% <= y <= 75%` chiều cao người).
- **`no-vest`**: Bounding box bao quanh vùng thân trên (torso) mặc quần áo dân dụng hoặc đồng phục không có dải phản quang an toàn.

---

## 4. Hợp Đồng Tiền Xử Lý (Crop Strategy Contract)

- **Person Detection & Padding**:
  - Từng cá nhân được phát hiện bởi mô hình Person Detector (COCO Class 0).
  - Vùng cắt (ROI) được mở rộng thêm **padding cố định 10px** (`person_roi_padding = 10`) để tránh cắt cụt mũ hoặc mép áo.
  - Vùng ảnh ROI được đưa về kích thước chuẩn $640 \times 640$ thông qua kỹ thuật **Letterbox Resize** (giữ nguyên tỷ lệ khung hình, bù đắp viền xám).

---

## 5. Chiến Lược Phân Chia Chống Rò Rỉ (Group-Aware Anti-Leakage Split)

Để ngăn chặn rò rỉ dữ liệu (Data Leakage) nghiêm trọng giữa các khung hình liên tiếp của cùng một video:
- **Group Key**: `camera_id + location_id + recording_session`.
- **Nguyên tắc bất biến**: Toàn bộ các khung hình thuộc cùng một phiên ghi hình (`recording_session`) chỉ được nằm trọn vẹn trong **DUY NHẤT 01 tập** (Train: 70%, Val: 15%, Test: 15%).
- **Hai giao thức kiểm thử độc lập**:
  - **Test A (Unseen Sessions)**: Đánh giá khả năng khái quát hóa theo thời gian và phiên làm việc khác nhau.
  - **Test B (Unseen Cameras)**: Tập Test chứa 02 góc camera độc lập (`cam_04`, `cam_05` tại `site_c`) hoàn toàn **chưa từng xuất hiện trong tập Train**, giúp đánh giá hiện tượng trôi miền dữ liệu (Domain Shift).
- **Khử trùng lặp**: Xác thực mã băm nội dung SHA-256 và Perceptual Hash (pHash) đảm bảo không có khung hình tương đồng cắt chéo tập.

Toàn bộ siêu dữ liệu 5,200 mẫu được theo dõi tại file [`split_manifest.csv`](./split_manifest.csv).
