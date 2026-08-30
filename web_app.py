"""Giao diện Web Tương tác (Streamlit Web Dashboard) cho Hệ thống PPE Surveillance.

Cho phép người dùng tải lên file ảnh/video, tùy chỉnh các ngưỡng phát hiện (Confidence, NMS, Frame Interval)
và xem trực tiếp kết quả phát hiện vi phạm trang bị bảo hộ kèm biểu đồ phân tích thống kê.

Chạy ứng dụng:
    streamlit run web_app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from ppe_detection.config import DetectionConfig
from ppe_detection.pipeline import PPEPipeline

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="PPE Safety Surveillance Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    """Inject CSS tùy biến để nâng cấp giao diện Web UI sang trọng và chuyên nghiệp."""
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #F8FAFC;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-left: 4px solid #3B82F6;
        }
        .metric-violation {
            border-left-color: #EF4444 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_custom_css()

    st.markdown("<div class='main-header'>🛡️ Hệ Thống Giám Sát Trang Bị Bảo Hộ AI (PPE)</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Ứng dụng Deep Learning YOLO & IoU Tracking phát hiện vi phạm mũ và áo bảo hộ thời gian thực</div>",
        unsafe_allow_html=True,
    )

    # 1. Thanh điều khiển Sidebar
    st.sidebar.header("⚙️ Cấu hình Hệ thống")

    demo_mode = st.sidebar.checkbox(
        "⚡ Chế độ Demo (Zero-Setup Mode)",
        value=True,
        help="Bật chế độ mô phỏng thử nghiệm mà không cần tải trước trọng số weights mô hình.",
    )

    col_m1, col_m2 = st.sidebar.columns(2)
    with col_m1:
        person_model_str = st.text_input("Model Person", value="models/yolov8n.pt", disabled=demo_mode)
    with col_m2:
        ppe_model_str = st.text_input("Model PPE", value="models/best.pt", disabled=demo_mode)

    st.sidebar.subheader("🎛️ Ngưỡng Phát Hiện")
    person_conf = st.sidebar.slider("Confidence Người", 0.1, 1.0, 0.3, 0.05)
    ppe_conf = st.sidebar.slider("Confidence PPE", 0.1, 1.0, 0.3, 0.05)
    confirm_frames = st.sidebar.slider("Khung hình xác nhận vi phạm", 1, 10, 2)
    detect_interval = st.sidebar.slider("Chạy detection sau N frame", 1, 10, 4)

    save_snapshots = st.sidebar.checkbox("📸 Lưu bằng chứng vi phạm (Snapshots)", value=True)

    # 2. Khu vực Tải dữ liệu Đầu vào
    st.markdown("### 📤 Tải lên Dữ liệu Giám sát")
    uploaded_file = st.file_uploader(
        "Chọn file Ảnh (.jpg, .png) hoặc Video (.mp4, .avi)",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    )

    if uploaded_file is None:
        st.info("💡 **Gợi ý**: Hãy tải lên một file ảnh hoặc video để trải nghiệm. Hoặc bật chế độ Demo ở thanh bên trái.")
        return

    # Lưu tạm file upload để OpenCV đọc
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # 3. Tiến hành Xử lý Dữ liệu
    st.markdown("---")
    st.markdown("### 🎬 Kết quả Xử lý & Phân tích")

    person_path = Path(person_model_str) if not demo_mode and person_model_str else None
    ppe_path = Path(ppe_model_str) if not demo_mode and ppe_model_str else None

    config = DetectionConfig(
        person_model_path=person_path,
        ppe_model_path=ppe_path,
        person_confidence=person_conf,
        ppe_confidence=ppe_conf,
        violation_confirmations=confirm_frames,
        detection_interval=detect_interval,
        show_window=False,
        save_output=True,
        save_snapshots=save_snapshots,
        output_dir=Path("outputs"),
        demo_mode=demo_mode,
        enable_beep=False,
    )

    with st.spinner("Đang thực thi nhận diện và theo dõi đối tượng..."):
        try:
            pipeline = PPEPipeline(config)
            report = pipeline.run(tmp_path)
        except Exception as err:
            st.error(f"❌ Xảy ra lỗi trong quá trình xử lý: {err}")
            return

    # 4. Hiển thị Thông số Thống kê Cards
    counts = report.counts
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Tổng Số Người Được Theo Dõi", len(report.unique_track_ids))
    col2.metric("⚠️ Số Người Vi Phạm", counts["people"])
    col3.metric("🪖 Vi Phạm Mũ Bảo Hộ", counts["helmet"])
    col4.metric("🦺 Vi Phạm Áo Phản Quang", counts["vest"])

    # 5. Hiển thị Kết quả Đầu ra (Ảnh hoặc Video kết quả)
    st.markdown("#### 🖼️ Xem Trực Quan Đầu Ra")
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        out_img_path = Path("outputs") / f"{Path(tmp_path).stem}_detected{suffix}"
        if out_img_path.exists():
            st.image(str(out_img_path), caption="Kết quả phát hiện PPE trên ảnh", use_container_width=True)
    else:
        out_vid_path = Path("outputs") / f"{Path(tmp_path).stem}_detected.mp4"
        if out_vid_path.exists():
            st.video(str(out_vid_path))

    # 6. Bảng Chi Tiết Sự Kiện Vi Phạm
    if report.events:
        st.markdown("#### 📋 Danh Sách Sự Kiện Vi Phạm Chi Tiết")
        df = pd.DataFrame([
            {
                "Track ID": e.track_id,
                "Loại Vi Phạm": "Mũ bảo hộ (Helmet)" if e.violation_type == "helmet" else "Áo phản quang (Vest)",
                "Frame": e.frame_id,
                "Thời điểm (Giây)": e.time_seconds,
                "Thời gian thực": e.detected_at,
                "Ảnh Bằng Chứng": e.snapshot_path,
            }
            for e in report.events
        ])
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
