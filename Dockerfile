FROM python:3.11-slim

# Cấu hình môi trường Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Cài đặt thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Sao chép và cài đặt các phụ thuộc Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn ứng dụng
COPY . .

# Expose cổng cho Streamlit Web Dashboard
EXPOSE 8501

# Lệnh khởi chạy ứng dụng mặc định
CMD ["streamlit", "run", "web_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
