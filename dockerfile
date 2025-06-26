# Use NVIDIA CUDA base image (Ubuntu-based)
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_DOCKER_ARCH=all
ENV GGML_CUDA=1

# Install build tools and dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopenblas-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libatlas-base-dev \
    gfortran \
    liblapack-dev \
    portaudio19-dev \
    autoconf \
    automake \
    libtool \
    ninja-build \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    ocl-icd-opencl-dev opencl-headers clinfo \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python3 -m pip install --upgrade pip setuptools wheel

# Set work directory
WORKDIR /app

# Copy requirements (excluding numpy/pandas/sklearn!)
COPY requirements.txt .

# Install safe packages first (except numpy, pandas, scikit-learn)
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# # Uninstall any cached/broken versions
# RUN pip uninstall -y numpy pandas scikit-learn || true

# # Install numpy FIRST from source (ensures compatibility with C extensions)
# RUN pip install --no-cache-dir --no-binary :all: "numpy<=2.2.0"

# # Now install pandas and scikit-learn from source so they link against the built numpy
# RUN pip install --no-cache-dir --no-binary :all: "pandas==2.2.2" "scikit-learn==1.5.0"

# # llama-cpp-python must come after numpy is stable
# ENV MAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"
# ENV CUDA_HOME=/usr/local/cuda
# Install llama-cpp-python with CUDA enabled

# Add symbolic links to CUDA stubs
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so \
 && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
 && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf \
 && ldconfig



# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir --force-reinstall llama-cpp-python


# Then downgrade numpy to a compatible version (e.g., 1.26.4)
RUN pip install numpy==1.26.4 --force-reinstall
RUN apt-get update && apt-get install -y ffmpeg


# RUN pip install uvicorn fastapi 
# Copy application source code
COPY . .

# Run the application
CMD ["python", "run.py"]
