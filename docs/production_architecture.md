# Kiến Trúc Triển Khai Production & Đánh Giá Giá Trị Nghiệp Vụ

Tài liệu này định hình kiến trúc mở rộng (Scale-out Architecture) cho hệ thống Giám sát Trang bị Bảo hộ Cá nhân (PPE) phục vụ 100+ người dùng đồng thời và kết nối đa camera công trường thời gian thực.

---

## 1. Kiến Trúc Hệ Thống Bất Đồng Bộ (Asynchronous Job Processing)

Khi đưa hệ thống ra môi trường Production với nhiều người dùng và camera, **KHÔNG** chạy suy luận AI (Inference) trực tiếp trong luồng giao diện Web (Streamlit/React). Hệ thống được tách biệt thành các dịch vụ độc lập:

```
[ Dashboard / React Client ]
           │ (HTTPS / REST API)
           ▼
     [ FastAPI Server ] ──────► [ PostgreSQL Database ] (Lưu trữ Metadata & Báo cáo)
           │
           │ (Push Job Request)
           ▼
     [ Redis Queue ] ──────────► [ Object Storage - MinIO/S3 ] (Lưu Video & Snapshots)
           │
           │ (Poll / Consume Jobs)
           ▼
 [ GPU Inference Worker Pool ]
 (NVIDIA TensorRT / Triton)
```

### Nguyên tắc vận hành:
1. **Khởi tạo duy nhất (Single Model Loading)**: Trọng số mô hình YOLO được nạp đúng 1 lần vào GPU VRAM khi GPU Worker khởi chạy.
2. **Xử lý Video Asynchronous**: Đăng tải video clip qua API `POST /api/v1/detect/video`, nhận ngay mã `job_id`.
3. **Cập nhật tiến độ**: Client theo dõi phần trăm hoàn thành qua WebSocket hoặc Long-polling API `GET /api/v1/jobs/{job_id}`.
4. **Giới hạn tài nguyên (Rate Limiting)**: Quản lý hàng đợi Redis giới hạn tối đa $N$ công việc đồng thời mỗi tài khoản.

---

## 2. Quản Lý Camera Real-time (RTSP Camera Stream Worker)

Để phục vụ hàng trăm camera IP công trường cùng lúc mà không nhân bản tải trọng mô hình:

```
[ RTSP Camera 01 ] ──┐
[ RTSP Camera 02 ] ──┼──► [ Camera Stream Worker (Single Instance per Camera) ]
[ RTSP Camera N  ] ──┘                 │
                                       ▼
                         [ Shared Event Stream (Kafka/Redis) ]
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
      [ PostgreSQL DB Log ]                     [ Live Dashboards (100 Users) ]
```

- **Mỗi Camera = 01 Worker Suy luận duy nhất**: 100 người dùng xem cùng 01 camera sẽ cùng đọc kết quả xử lý từ **Shared Event Stream**, không khởi tạo 100 instance model YOLO trùng lặp.
- **Tự động khôi phục (Auto-reconnect)**: Worker tự phục hồi kết nối luồng RTSP khi mất tín hiệu mạng mà không làm ngắt ngắt dịch vụ chung.

---

## 3. Quản Lý Caching & Đóng Gói State (`ModelBundle` vs `SessionPipeline`)

Phân tách rành mạch giữa dữ liệu trọng số dùng chung và trạng thái theo dõi đối tượng:

- **`ModelBundle` (Shared Resource)**: Chứa bộ weights YOLO Person & PPE đã nạp vào GPU/VRAM. Được cache toàn cục (`st.cache_resource` hoặc Singleton Service).
- **`SessionPipeline` (Isolated State)**: Chứa vết theo dõi `IoUTracker` và lịch sử xác nhận vi phạm (`confirmation_streaks`). Khởi tạo độc lập cho từng stream camera/session để tránh xung đột Track ID giữa các người dùng.

---

## 4. Bảo Mật, Quyền Riêng Tư & Tuân Thủ (Privacy & Security Compliance)

- **Xác thực & Phân quyền (Auth & RBAC)**: Phân quyền truy cập 3 cấp (Admin, Giám sát viên công trường, Viewer khách).
- **Quy tắc Lưu trữ Ảnh Snapshot (Retention Policy)**:
  - Tất cả snapshot vi phạm được tự động mã hóa AES-256 trên Object Storage.
  - Tự động xóa vĩnh viễn dữ liệu ảnh sau 30 ngày (Cron job cleanup).
  - Không lưu trữ hoặc truy xuất khuôn mặt đối tượng nếu use-case chỉ yêu cầu bảo hộ lao động.
- **Che mờ khuôn mặt (Face Blurring Option)**: Hỗ trợ tự động làm mờ vùng mặt khi trích xuất snapshot bằng chứng để tuân thủ luật bảo vệ dữ liệu cá nhân (GDPR/Bảo vệ dữ liệu cá nhân).

---

## 5. Giám Sát Vận Hành & Health Check (Monitoring & Alerting)

Hệ thống được theo dõi liên tục thông qua Prometheus & Grafana Dashboard:
- **FPS theo từng Camera**: Đảm bảo luồng xử lý $\ge 20$ FPS trên GPU.
- **Độ dài Hàng đợi (Redis Queue Depth)**: Phát hiện ùn tắc công việc.
- **GPU Utilization & Memory Usage**: Cảnh báo khi VRAM tràn quá $90\%$.
- **Model Inference Latency**: Thời gian phản hồi của từng khung hình.
- **Camera Offline Count**: Cảnh báo tức thì khi camera công trường bị đứt kết nối.

---

## 6. Kế Hoạch Đánh Giá Pilot Nghiệp Vụ (Business Validation Pilot)

Trước khi triển khai diện rộng trên toàn hệ thống công ty, tiến hành thử nghiệm Pilot:
- **Quy mô**: 01 công trường xây dựng, 03 camera quan sát vùng rủi ro cao, thời gian 02 tuần.
- **Quy trình xác minh**: Giám sát viên an toàn kiểm tra và đánh giá xác thực (Confirm / Reject) cho từng alert do AI phát ra.
- **Đo lường chỉ số ROI thực tế**:
  $$\text{Chỉ số Cải thiện An toàn} = \frac{\text{Số vi phạm được phát hiện và xử lý kịp thời}}{\text{Tổng thời gian giám sát thủ công tiết kiệm được (Giờ)}}$$
- Kết thúc 2 tuần thử nghiệm, thu thập feedback để điều chỉnh ngưỡng `conflict_margin` và `ppe_confidence` tối ưu nhất cho thực địa.
