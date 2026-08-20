# Phát hiện vi phạm trang bị bảo hộ bằng YOLO

Ứng dụng Computer Vision phát hiện người không đội mũ bảo hộ hoặc không mặc áo phản quang trong ảnh, video và luồng webcam. Hệ thống sử dụng kiến trúc hai giai đoạn: mô hình YOLO thứ nhất xác định vị trí người, sau đó mô hình YOLO PPE phân tích riêng từng vùng người để nhận diện trang bị bảo hộ.

> Đây là project học tập/portfolio. Kết quả không nên được dùng làm căn cứ duy nhất cho các quyết định liên quan đến an toàn lao động.

## Mục tiêu

- Tự động phát hiện người trong khu vực giám sát.
- Kiểm tra trạng thái mũ và áo bảo hộ của từng người.
- Gán ID tạm thời cho người xuất hiện trong video bằng IoU tracking.
- Cảnh báo âm thanh khi phát hiện một vi phạm mới.
- Hiển thị bounding box, nhãn PPE, độ tin cậy, FPS và thống kê vi phạm.
- Tổ chức mã nguồn theo module để dễ bảo trì và mở rộng.

## Tính năng

- Nhận đầu vào từ ảnh, video hoặc webcam.
- Tự động chọn CUDA, Apple MPS hoặc CPU.
- Phát hiện người bằng mô hình YOLO đã huấn luyện trên COCO.
- Phân loại PPE trên từng vùng người bằng model tùy chỉnh.
- Theo dõi đối tượng bằng chỉ số Intersection over Union (IoU).
- Hạn chế chạy inference ở mọi frame bằng `detection_interval`.
- Chỉ đếm một loại vi phạm một lần cho mỗi tracking ID.
- Cảnh báo bằng `winsound` trên Windows và terminal bell trên hệ điều hành khác.
- Kiểm tra đường dẫn model và tham số trước khi chạy.

## Kiến trúc xử lý

```text
Ảnh / Video / Webcam
        │
        ▼
YOLO Person Detector (class person của COCO)
        │
        ▼
Cắt vùng ROI của từng người
        │
        ▼
YOLO PPE Detector
        │
        ├── Helmet
        ├── No-Helmet
        ├── Vest
        └── No-Vest
        │
        ▼
IoU Tracker → Đếm vi phạm → Vẽ kết quả → Cảnh báo
```

Ultralytics tự thực hiện letterbox khi inference, vì vậy ảnh đầu vào không bị ép trực tiếp thành ảnh vuông và hạn chế hiện tượng méo tỉ lệ.

## Công nghệ sử dụng

- Python 3.10+
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- PyTorch
- OpenCV
- NumPy

## Cấu trúc source

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/                         # Tự tạo, không commit weight lớn
│   ├── yolov8n.pt                  # Model phát hiện người
│   └── best.pt                     # Model PPE tùy chỉnh
└── ppe_detection/
    ├── __init__.py
    ├── config.py                   # Dataclass cấu hình và validation
    ├── detector.py                 # Inference person và PPE
    ├── tracker.py                  # Theo dõi đối tượng bằng IoU
    ├── visualization.py            # Bounding box, nhãn và HUD
    └── pipeline.py                 # Điều phối ảnh/video/webcam
```

## Yêu cầu model

Project cần hai file weight:

1. **Person model:** model YOLO có class `person` ở class ID `0`, ví dụ `yolov8n.pt` được huấn luyện trên COCO.
2. **PPE model:** model tùy chỉnh có các nhãn tương ứng với `Helmet`, `No-Helmet`, `Vest` và `No-Vest`.

Tên class PPE được chuẩn hóa về chữ thường và chuyển `_` thành `-`. Nếu dataset sử dụng tên khác như `Hardhat`, `Safety Vest` hoặc `NO-Hardhat`, cần cập nhật logic ánh xạ trong `ppe_detection/detector.py`.

Không nên đưa model weight lớn vào Git. Có thể cung cấp model qua GitHub Releases, Google Drive hoặc hướng dẫn người dùng tự tải.

## Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Tạo môi trường ảo

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài thư viện

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Nếu sử dụng GPU NVIDIA, nên cài phiên bản PyTorch phù hợp với CUDA theo hướng dẫn trên trang chủ PyTorch trước khi cài các thư viện còn lại.

### 4. Chuẩn bị model

```text
models/
├── yolov8n.pt
└── best.pt
```

## Cách sử dụng

### Webcam mặc định

```bash
python app.py --source 0 --person-model models/yolov8n.pt --ppe-model models/best.pt
```

### File video

```bash
python app.py --source data/demo.mp4 --person-model models/yolov8n.pt --ppe-model models/best.pt
```

### File ảnh

```bash
python app.py --source data/test.jpg --person-model models/yolov8n.pt --ppe-model models/best.pt
```

### Tắt âm thanh cảnh báo

```bash
python app.py --source 0 --person-model models/yolov8n.pt --ppe-model models/best.pt --no-beep
```

### Điều chỉnh kích thước inference và chu kỳ detection

```bash
python app.py --source 0 --person-model models/yolov8n.pt --ppe-model models/best.pt --img-size 640 --detect-interval 4
```

Các tham số CLI:

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--source` | `0` | Camera index hoặc đường dẫn ảnh/video |
| `--person-model` | Bắt buộc | Đường dẫn model phát hiện người |
| `--ppe-model` | Bắt buộc | Đường dẫn model PPE |
| `--img-size` | `640` | Kích thước inference YOLO |
| `--detect-interval` | `4` | Chạy detection sau mỗi N frame |
| `--no-beep` | Tắt | Không phát cảnh báo âm thanh |

Nhấn `Esc` để dừng quá trình xử lý video hoặc webcam. Với ảnh, nhấn phím bất kỳ để đóng cửa sổ kết quả.

## Cấu hình mặc định

Các threshold nằm trong `DetectionConfig`:

| Cấu hình | Giá trị | Mô tả |
|---|---:|---|
| `person_confidence` | `0.3` | Confidence tối thiểu của người |
| `ppe_confidence` | `0.3` | Confidence tối thiểu của PPE |
| `nms_iou` | `0.5` | IoU threshold cho NMS của YOLO |
| `tracker_iou` | `0.3` | IoU tối thiểu để ghép detection với track |
| `max_disappeared` | `30` | Số lần cập nhật tối đa trước khi xóa track |

Threshold phù hợp phụ thuộc camera, góc nhìn, ánh sáng và dataset. Nên hiệu chỉnh bằng validation set thay vì chọn chỉ dựa trên hình ảnh demo.

## Quy tắc ghi nhận vi phạm

- `No-Helmet` được tính khi model phát hiện `No-Helmet` và không đồng thời phát hiện `Helmet` trên cùng ROI.
- `No-Vest` được tính tương tự.
- Mỗi cặp `(tracking ID, loại vi phạm)` chỉ được cộng một lần.
- `total` là tổng số **sự kiện theo loại vi phạm**, không nhất thiết là số người vi phạm. Một người thiếu cả mũ và áo làm `total` tăng hai lần.

Việc không phát hiện thấy `Helmet` chưa tự động đồng nghĩa với vi phạm; model cần nhận diện rõ class phủ định `No-Helmet`. Quy tắc này giúp hạn chế kết luận sai khi vật thể bị che khuất.

## Đánh giá mô hình

Khi trình bày project trong CV hoặc báo cáo, nên bổ sung các chỉ số đo trên test set độc lập:

- Precision, Recall và F1-score theo từng class.
- mAP@0.5 và mAP@0.5:0.95.
- Confusion matrix.
- FPS hoặc latency trên CPU/GPU cụ thể.
- Kết quả trong các điều kiện thiếu sáng, che khuất và góc camera khác nhau.

Không nên công bố số liệu ước lượng. Chỉ ghi các kết quả đã đo và mô tả rõ phần cứng, dữ liệu, image size và confidence threshold.

## Hạn chế hiện tại

- IoU tracker đơn giản có thể đổi ID khi người bị che khuất hoặc di chuyển nhanh.
- Bounding box giữa các frame inference được giữ lại, chưa có motion prediction.
- Chưa lưu ảnh/video kết quả ra file.
- Chưa có giao diện web, API hoặc dashboard thống kê.
- Độ chính xác PPE phụ thuộc mạnh vào chất lượng và phân bố dataset huấn luyện.
- Cảnh báo âm thanh chạy đồng bộ và có thể làm chậm vòng lặp trong thời gian rất ngắn.

## Hướng phát triển

- Thay IoU tracker bằng ByteTrack hoặc BoT-SORT.
- Thêm tùy chọn lưu video, ảnh và file CSV/JSON thống kê.
- Bổ sung vùng giám sát ROI và line-crossing.
- Thêm smoothing nhiều frame để giảm cảnh báo giả.
- Xây dựng REST API hoặc giao diện Streamlit/FastAPI.
- Export model sang ONNX/TensorRT để tối ưu triển khai.
- Thêm unit test cho IoU, tracker và quy tắc đếm vi phạm.
- Đóng gói bằng Docker và bổ sung GitHub Actions.

## Gợi ý mô tả trong CV

> Xây dựng hệ thống giám sát PPE hai giai đoạn bằng Ultralytics YOLO và OpenCV, phát hiện người rồi phân tích mũ/áo bảo hộ trên từng ROI; tích hợp IoU tracking, đếm vi phạm theo ID, cảnh báo thời gian thực và hỗ trợ CPU/GPU.

Hãy bổ sung các chỉ số thực tế của project, ví dụ số lượng ảnh trong dataset, mAP và FPS, trước khi đưa nội dung này vào CV.

## License

Chưa thiết lập license. Nếu public repository, hãy chọn license phù hợp và kiểm tra license của dataset, model weight, Ultralytics và các tài nguyên được sử dụng.
