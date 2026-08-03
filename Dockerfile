# ATOS Studio & Engine Unified Docker Container
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install build tools, OpenCV, Python, and Ninja
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libopencv-dev \
    python3 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    pyyaml \
    psutil \
    pydantic

WORKDIR /workspace
COPY . /workspace

# Configure and compile C++ Engine
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release

EXPOSE 8080 5005

CMD ["python3", "tools/web_gateway.py"]
