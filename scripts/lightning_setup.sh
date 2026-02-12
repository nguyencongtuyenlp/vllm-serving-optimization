#!/bin/bash
# Lightning.ai Setup Script for vLLM Serving Optimization
# Run this script in Lightning.ai Studio with GPU

set -e

echo "=========================================="
echo "vLLM Serving Optimization - Lightning.ai Setup"
echo "=========================================="
echo ""

# Step 1: Check GPU
echo "Step 1: Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✓ GPU detected!"
else
    echo "⚠ WARNING: nvidia-smi not found. Make sure you selected a GPU Studio."
    exit 1
fi
echo ""

# Step 2: Install Docker (if not already installed)
echo "Step 2: Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✓ Docker installed!"
else
    echo "✓ Docker already installed"
fi
echo ""

# Step 3: Install NVIDIA Container Toolkit
echo "Step 3: Installing NVIDIA Container Toolkit..."
if ! command -v nvidia-ctk &> /dev/null; then
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "✓ NVIDIA Container Toolkit installed!"
else
    echo "✓ NVIDIA Container Toolkit already installed"
fi
echo ""

# Step 4: Clone or navigate to project
echo "Step 4: Setting up project..."
if [ ! -d "vllm" ]; then
    echo "Project directory not found. Please clone your repo first:"
    echo "  git clone <your-repo-url> vllm"
    echo "  cd vllm"
    echo "  bash scripts/lightning_setup.sh"
    exit 1
fi

cd vllm
echo "✓ In project directory: $(pwd)"
echo ""

# Step 5: Setup environment
echo "Step 5: Setting up environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ Created .env from template"
else
    echo "✓ .env already exists"
fi
echo ""

# Step 6: Install Python dependencies
echo "Step 6: Installing Python dependencies..."
pip install -q -r requirements.txt
echo "✓ Python dependencies installed"
echo ""

# Step 7: Start vLLM server
echo "Step 7: Starting vLLM server..."
docker compose up -d
echo "✓ Server starting in background..."
echo ""

# Step 8: Wait for server to be ready
echo "Step 8: Waiting for server to be ready..."
max_attempts=60
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        break
    fi
    
    attempt=$((attempt + 1))
    if [ $((attempt % 10)) -eq 0 ]; then
        echo "  Waiting... ($attempt/$max_attempts)"
    fi
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "✗ Server failed to start within timeout"
    echo "Check logs with: docker compose logs"
    exit 1
fi
echo ""

# Step 9: Test server
echo "Step 9: Testing server..."
curl -s http://localhost:8000/v1/models | python -m json.tool
echo ""
echo "✓ Server is responding!"
echo ""

# Step 10: Instructions
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run quick benchmark:"
echo "   python -m bench.benchmark --profile quick"
echo ""
echo "2. Run full benchmark suite:"
echo "   python -m bench.benchmark --profile light"
echo "   python -m bench.benchmark --profile medium"
echo ""
echo "3. View results:"
echo "   ls -lh results/"
echo "   cat results/run_*.csv"
echo ""
echo "4. View server logs:"
echo "   docker compose logs -f"
echo ""
echo "5. Stop server when done:"
echo "   docker compose down"
echo ""
