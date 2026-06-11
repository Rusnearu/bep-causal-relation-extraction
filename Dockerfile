# Base image: PyTorch 2.6.0 with CUDA 12.4 and Python 3.11
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system packages needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    perl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        -r requirements.txt \
    && pip install --no-cache-dir jupyter nbconvert

# Copy the full repository into the image
COPY . .

# Jupyter port
EXPOSE 8888

# Default: start Jupyter so notebooks can be opened in a browser
CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token="]
