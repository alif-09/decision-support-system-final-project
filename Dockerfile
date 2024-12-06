# Gunakan image Python resmi
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Salin file aplikasi ke container
COPY . /app

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Ekspos port untuk Flask
EXPOSE 8080

# Command untuk menjalankan aplikasi
CMD ["python", "app.py"]
