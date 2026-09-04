# Quy Tắc Nghiệp Vụ & Tiêu Chí Thành Công — PPE Surveillance

## 1. Bài Toán Nghiệp Vụ Có Thể Đo Lường (Problem Statement)

Hệ thống tự động phát hiện người công nhân không đội mũ bảo hộ (`no-helmet`) hoặc không mặc áo phản quang (`no-vest`) trong hình ảnh và video giám sát tại công trường xây dựng/nhà máy. 

Hệ thống thực hiện:
- Gán mã định danh tạm thời (Track ID) cho từng cá nhân trong tầm quan sát.
- Phát hiện và kích hoạt cảnh báo khi tình trạng vi phạm diễn ra liên tiếp trong ít nhất $N$ khung hình ($N \ge 2$).
- Tự động trích xuất và lưu trữ hình ảnh bằng chứng (ROI snapshot) cùng báo cáo vi phạm phục vụ công tác giám sát an toàn lao động.

### Phạm vi thông số vận hành tiêu chuẩn:
- **Đối tượng giám sát**: Tất cả nhân sự/người xuất hiện trong khung hình thuộc vùng nguy hiểm ROI.
- **Thiết bị đầu vào**: Camera IP cố định (chuẩn kết nối RTSP), Video clip giám sát, hoặc Webcam trực tiếp.
- **Điều kiện môi trường**: Ánh sáng ban ngày hoặc hệ thống chiếu sáng công nghiệp đạt tối thiểu 150 lux; khoảng cách quan sát từ 3m đến 25m.
- **Bảo hộ kiểm tra (PPE)**: Mũ bảo hộ (`helmet` / `no-helmet`) và Áo phản quang (`vest` / `no-vest`).
- **Thời gian phản hồi (Time-to-Alert)**: $\le 2.0$ giây kể từ khi vi phạm xuất hiện.
- **Tần suất cảnh báo sai (False Alarm Rate)**: $\le 2$ lần/giờ/camera.
- **Bảo mật**: Không lưu trữ dữ liệu khuôn mặt nhận dạng cá nhân; ảnh bằng chứng (snapshot) được lưu trữ tối đa 30 ngày.

---

## 2. Tiêu Chí Thành Công (Success Criteria)

| Tiêu chí | Chỉ số Mục tiêu Ban đầu | Phương pháp Đo lường |
| :--- | :--- | :--- |
| **Recall (Mũ bảo hộ - `no-helmet`)** | $\ge 90\%$ | Đo trên tập dữ liệu Test độc lập (Test Set) |
| **Recall (Áo phản quang - `no-vest`)** | $\ge 85\%$ | Đo trên tập dữ liệu Test độc lập (Test Set) |
| **Precision Cảnh báo (Event Precision)** | $\ge 85\%$ | Tỷ lệ sự kiện vi phạm xác nhận đúng / Tổng số sự kiện phát ra |
| **Tần suất Cảnh báo Sai (False Alerts)** | $\le 2$ lần/giờ/camera | Thử nghiệm trên video thực tế 24h |
| **Thời gian Cảnh báo (Time-to-Alert)** | $\le 2.0$ giây | Độ trễ từ frame vi phạm đầu tiên đến khi phát cảnh báo |
| **Tốc độ Xử lý GPU (Nvidia RTX/T4)** | $\ge 20$ FPS | Tốc độ suy luận hai giai đoạn kèm IoU Tracking |
| **Tốc độ Xử lý CPU (Intel i7/Xeon)** | $\ge 5$ FPS | Tốc độ suy luận ở chế độ tối ưu CPU |

> [!NOTE]
> Bảng tiêu chí trên là chỉ số mục tiêu thiết kế hệ thống. Chỉ số chính thức được cập nhật sau khi đánh giá trên tập dữ liệu Test độc lập.

---

## 3. Quy Tắc Ghi Nhận Vi Phạm Chuẩn (Standard Business Rules)

1. **Vùng Nguy Hiểm (ROI Check)**:
   Chỉ kiểm tra và ghi nhận vi phạm đối với những đối tượng có điểm tâm Bounding Box ($X_{center}, Y_{center}$) nằm hoàn toàn bên trong vùng nguy hiểm (ROI Polygon).

2. **Xác Nhận Vi Phạm & Xử Lý Xung Đột Nhãn**:
   - Nhãn `no-helmet` hoặc `no-vest` được tính khi confidence $\ge \text{ppe\_confidence}$.
   - Nếu đồng thời xuất hiện nhãn tuân thủ (`helmet` / `vest`), vi phạm chỉ được xác nhận khi:
     $$\text{Confidence}(\text{no-helmet}) > \text{Confidence}(\text{helmet}) + \text{conflict\_margin} \quad (\text{với } \text{conflict\_margin} = 0.1)$$
   - Sự kiện vi phạm chính thức được kích hoạt sau khi tình trạng vi phạm xuất hiện liên tiếp trong ít nhất $N$ lần inference.

3. **Giới Hạn Sự Kiện Vi Phạm (Unique Event per Session)**:
   Một Track ID duy nhất chỉ tạo tối đa 01 sự kiện cảnh báo chính thức cho mỗi loại vi phạm (`helmet` hoặc `vest`) trong cùng một phiên theo dõi, nhằm tránh gây trần ngập thông báo trùng lặp.

4. **Xử Lý Thiếu Nhãn (Missing Detection)**:
   Mô hình không phát hiện thấy `helmet` **KHÔNG** tự động đồng nghĩa với vi phạm `no-helmet`. Tình trạng thiếu nhãn nhận diện sẽ được coi là không đủ căn cứ kết luận vi phạm.

5. **Tái Khởi Tạo Track (Re-identification After ID Loss)**:
   Nếu một đối tượng bị mất dấu (Disappeared $> \text{max\_disappeared}$ frames) và xuất hiện lại với một Track ID mới, đối tượng mới này sẽ được xử lý như một phiên theo dõi độc lập.

---

## 4. Trường Hợp Ngoài Phạm Vi (Out of Scope)

Hệ thống phiên bản hiện tại **CHƯA** hỗ trợ xử lý các trường hợp sau:
- Mũ bảo hộ cầm trên tay hoặc treo bên hông thay vì đội trên đầu.
- Áo phản quang bị che khuất quá $60\%$ diện tích bởi vật dụng hoặc tư thế đứng.
- Đối tượng công nhân ở quá xa camera (kích thước Bounding Box $< 32 \times 32$ pixels).
- Khung hình từ camera bị rung lắc quá mạnh do thời tiết hoặc va đập cơ học.
- Đám đông quá 10 người đứng chồng lấn mật độ cao (Occlusion $> 70\%$).
- Mũ trang bị không thuộc tiêu chuẩn an toàn lao động (ví dụ: mũ lưỡi trai, mũ cối, nón lá).
- Nhận dạng danh tính cá nhân (Face Recognition / Employee ID Matching).
