# Use an official Python base image
FROM python:3.10-slim


# Set work directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install faiss-cpu via pip
RUN pip install --no-cache-dir faiss-cpu

# Copy your requirements (if you have one)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Expose port (match with `serve(..., port=5001)`)
EXPOSE 5001

# Start the app using waitress
CMD ["python", "app.py"]
