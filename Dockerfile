# Gunakan base image ringan
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Salin file ke dalam container
COPY api/ api/
COPY models/ models/
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan API
RUN mkdir -p /app/logs
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]