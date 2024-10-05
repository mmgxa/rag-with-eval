# Stage 1: Build
FROM python:3.11-slim 

# Install build dependencies
RUN export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
  build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies in a clean layer
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY . .

CMD ["streamlit", "run", "app.py"]