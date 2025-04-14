FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies (e.g., wget, curl, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    libncurses5-dev \
    libssl-dev \
    libffi-dev \
    software-properties-common \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Add repository for Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update
# Install Python 3.11
RUN apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils

# Set Python 3.11 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
# Install pytorch
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace
# Run script when container starts
# Copy and install
COPY requirements.txt /workspace
RUN pip install -r requirements.txt

# Install cudnn
#RUN apt-get update -y && apt-get install -y python3 python3-pip libcudnn8 libcudnn8-dev

# Set CUDA paths
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda-11.8/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda-11.8

# Copy the rest
COPY . /workspace

# Run FastAPI app
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]