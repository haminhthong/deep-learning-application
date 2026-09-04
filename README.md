# PPE Safety Surveillance - Phát hiện vi phạm bảo hộ bằng YOLO

Hệ thống Computer Vision phát hiện người không đội mũ bảo hộ hoặc không mặc áo phản quang trong ảnh, video và webcam. Project sử dụng hai mô hình YOLO theo chuỗi, IoU Tracker, cơ chế xác nhận vi phạm qua nhiều lần phát hiện, lưu ảnh bằng chứng và xuất báo cáo JSON/CSV.

> Project phục vụ học tập và portfolio. Kết quả không nên là căn cứ duy nhất cho quyết định liên quan đến an toàn lao động.

## Bài toán

Giám sát PPE trực tiếp trên toàn khung hình thường gặp hai khó khăn: vật thể bảo hộ nhỏ so với ảnh và một người có thể bị đếm vi phạm nhiều lần qua các frame. Project giải quyết theo hai giai đoạn:

1. YOLO phát hiện người trên toàn khung hình.
2. Mỗi vùng người được cắt riêng và đưa qua model PPE.
3. IoU Tracker duy trì ID tạm thời giữa các lần phát hiện.
4. Vi phạm phải xuất hiện liên tiếp đủ số lần cấu hình mới được ghi nhận.
5. Hệ thống trực quan hóa, cảnh báo và xuất bằng chứng/báo cáo.

## Tính năng nổi bật

- Nhận đầu vào từ ảnh, video hoặc webcam.
- Hai chế độ: inference bằng model thật và demo mô phỏng không cần weight.
- Tự chọn CUDA, Apple MPS hoặc CPU.
- Phát hiện `helmet`, `no-helmet`, `vest`, `no-vest` trên từng người.
- Theo dõi ID bằng IoU và giữ track khi đối tượng mất dấu tạm thời.
- Xác nhận vi phạm qua nhiều lần inference để giảm cảnh báo giả.
- Tùy chọn vùng giám sát ROI dạng polygon.
- Lưu ảnh/video đã chú thích.
- Lưu snapshot người vi phạm làm bằng chứng.
- Xuất báo cáo JSON tổng hợp và CSV sự kiện.
- Chạy không giao diện trên server với `--no-display`.
- Giao diện web Streamlit cho ảnh/video upload.
- Unit test cho cấu hình, tracker, reporting và demo pipeline.

## Kiến trúc

```text
Ảnh / Video / Webcam
        |
        v
YOLO Person Detector (COCO class 0)
        |
        v
Cắt ROI của từng người
        |
        v
YOLO PPE Detector
        |
        +-- helmet / no-helmet
        +-- vest / no-vest
        |
        v
IoU Tracker
        |
        v
Xác nhận nhiều frame
        |
        +-- Hiển thị HUD
        +-- Cảnh báo âm thanh
        +-- Snapshot bằng chứng
        +-- JSON / CSV / Video đầu ra
```

Ultralytics thực hiện letterbox khi inference nên ảnh không bị ép méo trực tiếp thành hình vuông.

## Công nghệ

- Python 3.10+
- Ultralytics YOLO
- PyTorch
- OpenCV
- NumPy
- Streamlit và Pandas
- Pytest

## Cấu trúc project

```text
deep-learning-application/
├── app.py                         # Giao diện dòng lệnh
├── web_app.py                     # Dashboard Streamlit
├── requirements.txt
├── pytest.ini
├── LICENSE
├── README.md
├── ppe_detection/
│   ├── config.py                  # Cấu hình và kiểm tra tham số
│   ├── models.py                  # Dataclass dữ liệu nghiệp vụ
│   ├── detector.py                # YOLO detector và mock detector
│   ├── tracker.py                 # IoU Tracker
│   ├── pipeline.py                # Điều phối toàn bộ luồng xử lý
│   ├── reporting.py               # Báo cáo JSON/CSV
│   └── visualization.py           # Bounding box, ROI và HUD
└── tests/                          # Kiểm thử tự động
```

## Yêu cầu model

Chế độ thật cần hai file weight:

1. Model phát hiện người có class `person` tại class ID `0`, ví dụ `yolov8n.pt` huấn luyện trên COCO.
2. Model PPE có các class `helmet`, `no-helmet`, `vest`, `no-vest`.

Tên class được chuẩn hóa chữ thường, dấu cách và `_` thành `-`. Nếu dataset dùng tên như `Hardhat` hoặc `Safety Vest`, hãy sửa ánh xạ trong `ppe_detection/detector.py`.

Không commit weight hoặc dữ liệu lớn vào Git. Nên cung cấp qua GitHub Releases hoặc đường dẫn tải riêng.

## Cài đặt

### Windows PowerShell

```powershell
git clone <repository-url>
cd deep-learning-application
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS

```bash
git clone <repository-url>
cd deep-learning-application
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu dùng GPU NVIDIA, hãy cài bản PyTorch tương thích CUDA theo hướng dẫn chính thức của PyTorch.

## Chạy nhanh không cần model

Chế độ demo tạo hai người mô phỏng để kiểm tra pipeline, tracking, giao diện và output. Đây không phải kết quả inference thật.

```bash
python app.py --demo --source data/test.jpg --save --no-display
```

## Chạy bằng model thật

Chuẩn bị:

```text
models/
├── yolov8n.pt
└── best.pt
```

### Webcam

```bash
python app.py --source 0 --person-model models/yolov8n.pt --ppe-model models/best.pt
```

### Ảnh

```bash
python app.py --source data/test.jpg --person-model models/yolov8n.pt --ppe-model models/best.pt --save
```

### Video không mở cửa sổ

```bash
python app.py --source data/demo.mp4 --person-model models/yolov8n.pt --ppe-model models/best.pt --save --no-display
```

Nhấn `Esc` để dừng webcam/video khi cửa sổ OpenCV đang mở.

## Giao diện web

```bash
streamlit run web_app.py
```

Dashboard cho phép upload ảnh/video, chọn demo hoặc model thật, điều chỉnh confidence, chu kỳ inference và số lần xác nhận vi phạm.

## Tham số CLI

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--source` | `0` | Chỉ số camera hoặc đường dẫn ảnh/video |
| `--person-model` | trống | Model phát hiện người |
| `--ppe-model` | trống | Model PPE |
| `--demo` | tắt | Dùng dữ liệu phát hiện mô phỏng |
| `--img-size` | `640` | Kích thước inference YOLO |
| `--detect-interval` | `4` | Chạy detector sau mỗi N frame |
| `--person-conf` | `0.3` | Confidence tối thiểu của người |
| `--ppe-conf` | `0.3` | Confidence tối thiểu của PPE |
| `--confirm-frames` | `2` | Số lần phát hiện liên tiếp để xác nhận |
| `--save` | tắt | Lưu output, báo cáo và snapshot |
| `--no-snapshots` | tắt | Không lưu ảnh bằng chứng |
| `--output-dir` | `outputs` | Thư mục đầu ra |
| `--no-display` | tắt | Không mở cửa sổ OpenCV |
| `--no-beep` | tắt | Không phát âm cảnh báo |

Nếu không truyền đủ hai model, CLI tự chuyển sang demo và ghi cảnh báo rõ ràng.

## Quy tắc vi phạm

- `no-helmet` được tính khi model phát hiện nhãn này và không đồng thời phát hiện `helmet` trên cùng ROI.
- `no-vest` được xử lý tương tự.
- Không phát hiện thấy PPE không tự động được coi là vi phạm.
- Mỗi cặp `(track_id, loại_vi_phạm)` chỉ tạo một sự kiện trong phiên.
- Với video, vi phạm phải xuất hiện đủ `confirm-frames` lần inference liên tiếp.
- Với ảnh tĩnh, kết quả được ghi nhận ngay vì chỉ có một frame.
- `total` là tổng số sự kiện theo loại; một người thiếu cả mũ và áo tạo hai sự kiện.

## Output

Khi bật `--save`, thư mục output có thể gồm:

```text
outputs/
├── demo_detected.mp4
├── demo_report.json
├── demo_events.csv
└── snapshots/
    └── violation_id2_helmet_frame24_....jpg
```

JSON chứa thông tin phiên, tổng frame, số ID từng theo dõi, tổng hợp vi phạm và danh sách sự kiện. CSV thuận tiện cho Excel hoặc dashboard phân tích.

## Kiểm thử

```bash
pytest -q
```

Các test hiện có kiểm tra:

- Giá trị cấu hình và validation.
- IoU, duy trì ID và xóa track.
- Đếm sự kiện, xuất JSON/CSV.
- Chạy end-to-end ảnh tổng hợp bằng mock detector.

## Đánh giá để đưa vào CV

Không nên tự điền số liệu ước lượng. Hãy đo trên test set độc lập và bổ sung:

- Số ảnh và phân bố từng class trong dataset.
- Precision, Recall, F1-score theo class.
- mAP@0.5 và mAP@0.5:0.95.
- FPS/latency, GPU hoặc CPU, image size và batch size.
- Tỉ lệ cảnh báo giả trước/sau cơ chế xác nhận nhiều frame.
- Kết quả trong điều kiện thiếu sáng, che khuất và góc camera khác nhau.

Mẫu mô tả CV:

> Xây dựng hệ thống giám sát PPE hai giai đoạn bằng Ultralytics YOLO và OpenCV; phát hiện người, phân tích mũ/áo trên từng ROI, duy trì ID bằng IoU Tracker, xác nhận vi phạm nhiều frame, lưu ảnh bằng chứng và xuất báo cáo JSON/CSV; hỗ trợ CLI, Streamlit, CPU/GPU và chế độ headless.

## Hạn chế

- IoU Tracker có thể đổi ID khi đối tượng bị che khuất lâu hoặc di chuyển nhanh.
- Bounding box được giữ nguyên giữa các frame không chạy detector, chưa có motion prediction.
- Logic PPE phụ thuộc đúng taxonomy class của model huấn luyện.
- Mock demo chỉ kiểm tra luồng phần mềm, không đại diện cho độ chính xác AI.
- Chưa có authentication, database hoặc triển khai multi-camera.

## Hướng phát triển

- Thay IoU Tracker bằng ByteTrack hoặc BoT-SORT.
- Dùng voting theo thời gian và confidence calibration.
- Thêm cấu hình ROI từ giao diện thay vì khai báo trong code.
- Export ONNX/TensorRT cho edge deployment.
- Thêm REST API, Docker và CI/CD.
- Lưu sự kiện vào PostgreSQL và xây dashboard theo ca/khu vực.
- Bổ sung benchmark tái lập và model card.

## License

Source code phát hành theo MIT License. Người dùng vẫn cần kiểm tra license của Ultralytics, dataset và model weight trước khi sử dụng thương mại.
