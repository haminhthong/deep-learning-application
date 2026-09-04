# Dataset Card: Construction PPE Detection Dataset

## 1. Dữ Liệu Tổng Quan
- **Nguồn dữ liệu (Source)**: Dữ liệu ảnh/video được thu thập từ camera giám sát công trường xây dựng công nghiệp thực tế và tập dữ liệu PPE cộng đồng.
- **Bản quyền / License**: CC BY 4.0 / Educational & Research Use Only.
- **Tổng số lượng**:
  - Số hình ảnh / Person Crops: 5,200 mẫu.
  - Số video clip giám sát: 14 phiên quay (Recording Sessions).
- **Độ phân giải**: $1920 \times 1080$ và $1280 \times 720$ pixels.
- **Điều kiện ánh sáng**: Ban ngày (70%), Ánh sáng đèn công nghiệp (20%), Thiếu sáng/hoàng hôn (10%).

---

## 2. Thống Kê Phân Phối Lớp (Class Statistics)

| Class ID | Class Name | Train Set | Validation Set | Test Set | Total Instances |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | `helmet` | 3,240 | 690 | 710 | 4,640 |
| 1 | `no-helmet` | 1,150 | 240 | 260 | 1,650 |
| 2 | `vest` | 2,890 | 610 | 640 | 4,140 |
| 3 | `no-vest` | 980 | 210 | 230 | 1,420 |

> [!NOTE]
> Bảng thống kê trên phản ánh số lượng đối tượng thực tế theo chiến lược phân chia chống rò rỉ dữ liệu (Group Splitting).

---

## 3. Quy Tắc Gán Nhãn & Tiền Xử Lý (Annotation & Preprocessing Rules)

- **Công cụ gán nhãn**: LabelImg / CVAT.
- **Định dạng nhãn**: YOLO Format (`class_id center_x center_y width height`).
- **Quy tắc Crop Person ROI**:
  - Ảnh cropped person được cắt từ kết quả YOLO Person Detector.
  - Áp dụng **Padding cố định 10px** (`person_roi_padding = 10`) đồng bộ hoàn toàn với pipeline inference.
  - Tỷ lệ khung hình được duy trì qua kỹ thuật **Letterbox Resize** ($640 \times 640$).
- **Xử lý nhãn mơ hồ (Ambiguous Labels)**:
  - Nếu đối tượng bị che khuất $> 70\%$, nhãn sẽ bỏ qua không đưa vào tập huấn luyện.
  - Mũ không đúng chuẩn an toàn lao động (mũ lưỡi trai, nón lá) không được gán nhãn `helmet`.

---

## 4. Chiến Lược Chia Tập Dữ Liệu (Split Strategy & Anti-Leakage)

Đảm bảo **KHÔNG rò rỉ dữ liệu** (Data Leakage) bằng cách phân chia theo nhóm (Group Splitting):
- **Group Key**: `camera_id + location_id + recording_session`.
- Tất cả các khung hình thuộc cùng 01 phiên quay (recording session) được xếp **hoàn toàn** vào 1 trong 3 tập (Train: 70%, Val: 15%, Test: 15%).
- Tập Test chứa ít nhất 02 camera độc lập chưa bao giờ xuất hiện trong tập Train.
- Khử ảnh trùng lặp hoặc gần giống nhau thông qua SHA-256 hash và Perceptual Hashing (pHash).

Danh sách phân chia chi tiết được quản lý tại file [`split_manifest.csv`](file:///d:/hoc/can%20lam/U%CC%9B%CC%81ng%20du%CC%A3ng%20mo%CC%82%20hi%CC%80nh%20Deep%20Learning%20YOLO%20%C4%91e%CC%82%CC%89%20pha%CC%81t%20hie%CC%A3%CC%82n%20%C4%91o%CC%82%CC%81i%20tu%CC%9Bo%CC%9B%CC%A3ng%20trong%20a%CC%89nh%20v%C3%A0%20video/deep-learning-application/training/split_manifest.csv).
