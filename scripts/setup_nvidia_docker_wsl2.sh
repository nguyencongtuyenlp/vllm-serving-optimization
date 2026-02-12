#!/bin/bash
# Setup NVIDIA Container Toolkit for WSL2 Ubuntu
# Run this script inside WSL2 Ubuntu

set -e

echo "==================================="
echo "NVIDIA Container Toolkit Setup for WSL2"
echo "==================================="

# Check if running in WSL2
if ! grep -qi microsoft /proc/version; then
    echo "ERROR: This script must be run inside WSL2"
    exit 1
fi

echo ""
echo "Step 1: Cleaning up old configs..."
sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo ""
echo "Step 2: Adding NVIDIA Container Toolkit repository (generic deb)..."

# Use the generic deb repository URL
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add the stable deb repository
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo ""
echo "Step 3: Updating package list..."
sudo apt-get update

echo ""
echo "Step 4: Installing NVIDIA Container Toolkit..."
sudo apt-get install -y nvidia-container-toolkit

echo ""
echo "Step 5: Configuring Docker to use NVIDIA runtime..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "WARNING: Docker not found in WSL2."
    echo "Make sure Docker Desktop is installed on Windows and WSL2 integration is enabled."
    echo ""
    echo "To enable WSL2 integration:"
    echo "1. Open Docker Desktop on Windows"
    echo "2. Go to Settings > Resources > WSL Integration"
    echo "3. Enable integration with Ubuntu"
    echo "4. Click 'Apply & Restart'"
    echo ""
    echo "After that, restart WSL2 and run this script again."
    exit 1
fi

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

echo ""
echo "Step 6: Restarting Docker..."

# In WSL2, Docker is managed by Docker Desktop on Windows
# We can't use systemctl, so we just notify the user
echo "NOTE: In WSL2, Docker is managed by Docker Desktop."
echo "Please restart Docker Desktop on Windows to apply changes:"
echo "1. Right-click Docker Desktop icon in system tray"
echo "2. Click 'Restart'"

echo ""
echo "==================================="
echo "✓ Installation completed!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Restart Docker Desktop on Windows"
echo "2. Test GPU access with:"
echo "   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
echo ""
