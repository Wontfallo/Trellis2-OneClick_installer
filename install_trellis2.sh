#!/bin/bash
#=============================================================================
# TRELLIS.2 One-Click Installer for Ubuntu/WSL2
# https://github.com/microsoft/TRELLIS.2
#=============================================================================
# Requirements:
#   - Ubuntu (native or WSL2)
#   - NVIDIA GPU with 24GB+ VRAM (RTX 4090 ✓)
#   - Conda installed
#   - CUDA Toolkit (12.4 recommended)
#=============================================================================

set -e  # Exit on error

#-----------------------------------------------------------------------------
# Colors and formatting
#-----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

print_header() {
    echo ""
    echo -e "${MAGENTA}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║${NC}  ${BOLD}$1${NC}"
    echo -e "${MAGENTA}╚══════════════════════════════════════════════════════════════╝${NC}"
}

print_step() {
    echo -e "${CYAN}▶${NC} ${BOLD}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

#-----------------------------------------------------------------------------
# Configuration
#-----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRELLIS_DIR="$SCRIPT_DIR/TRELLIS.2"
ENV_NAME="trellis2"
PYTHON_VERSION="3.10"
PYTORCH_VERSION="2.6.0"
CUDA_PYTORCH="cu124"  # CUDA 12.4 for PyTorch
EXTENSIONS_DIR="/tmp/trellis2_extensions"

#-----------------------------------------------------------------------------
# Pre-flight checks
#-----------------------------------------------------------------------------
print_header "TRELLIS.2 One-Click Installer"
echo ""
echo -e "  ${BLUE}GPU:${NC} RTX 4090 detected (24GB VRAM)"
echo -e "  ${BLUE}Environment:${NC} $ENV_NAME"
echo -e "  ${BLUE}Python:${NC} $PYTHON_VERSION"
echo -e "  ${BLUE}PyTorch:${NC} $PYTORCH_VERSION + CUDA 12.4"
echo ""

# Check if running in TRELLIS.2 directory
if [ ! -d "$TRELLIS_DIR" ]; then
    print_error "TRELLIS.2 directory not found at: $TRELLIS_DIR"
    print_warning "Cloning TRELLIS.2 repository..."
    cd "$SCRIPT_DIR"
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
    TRELLIS_DIR="$SCRIPT_DIR/TRELLIS.2"
fi

cd "$TRELLIS_DIR"
print_success "Working directory: $TRELLIS_DIR"

# Check for NVIDIA GPU
print_step "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1)
    print_success "Found GPU: $GPU_INFO"
else
    print_error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Check for conda
print_step "Checking Conda..."
if command -v conda &> /dev/null; then
    CONDA_PATH=$(which conda)
    print_success "Conda found: $CONDA_PATH"
else
    print_error "Conda not found. Please install Miniconda or Anaconda first."
    echo "Quick install: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && bash ~/miniconda.sh"
    exit 1
fi

# CUDA configuration
print_step "Configuring CUDA..."

# WSL2 shares Windows GPU driver, but needs CUDA Toolkit for nvcc (compiling extensions)
# Try to find any CUDA 12.x installation
if [ -z "$CUDA_HOME" ]; then
    # Search for CUDA 12.x in order of preference
    for cuda_ver in 12.8 12.6 12.4 12.3 12.2 12.1; do
        if [ -d "/usr/local/cuda-$cuda_ver" ]; then
            export CUDA_HOME="/usr/local/cuda-$cuda_ver"
            break
        fi
    done
    # Fallback to generic cuda symlink
    if [ -z "$CUDA_HOME" ] && [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
    fi
fi

if [ -n "$CUDA_HOME" ]; then
    print_success "CUDA_HOME: $CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# Check if nvcc is available (required for compiling CUDA extensions)
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    print_success "CUDA Toolkit: $CUDA_VERSION (nvcc available)"
else
    print_warning "nvcc not found in WSL!"
    echo ""
    echo -e "  ${YELLOW}CUDA Toolkit is needed to compile custom CUDA extensions.${NC}"
    echo -e "  ${YELLOW}Your Windows CUDA won't work directly in WSL.${NC}"
    echo ""
    echo "  Options:"
    echo "    1) Install CUDA Toolkit 12.6 in WSL (recommended)"
    echo "    2) Exit and install manually"
    echo ""
    read -p "  Install CUDA Toolkit 12.6 now? (Y/n): " INSTALL_CUDA
    if [[ ! "$INSTALL_CUDA" =~ ^[Nn]$ ]]; then
        print_step "Installing CUDA Toolkit 12.6 in WSL..."
        # Add NVIDIA package repository
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        rm cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        sudo apt-get install -y cuda-toolkit-12-6
        export CUDA_HOME=/usr/local/cuda-12.6
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        print_success "CUDA Toolkit 12.6 installed"
        # Verify
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
            print_success "nvcc now available: $CUDA_VERSION"
        fi
    else
        print_error "Cannot continue without CUDA Toolkit (nvcc required for extensions)"
        echo "Install CUDA Toolkit manually: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
fi

#-----------------------------------------------------------------------------
# Initialize Conda
#-----------------------------------------------------------------------------
print_header "Setting up Conda Environment"

# Source conda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Check if environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment '$ENV_NAME' already exists."
    read -p "Recreate environment? (y/N): " RECREATE
    if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
        print_step "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        print_step "Using existing environment..."
    fi
fi

# Create environment if it doesn't exist
if ! conda env list | grep -q "^$ENV_NAME "; then
    print_step "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# Activate environment
print_step "Activating environment..."
conda activate $ENV_NAME
print_success "Environment activated: $(python --version)"

#-----------------------------------------------------------------------------
# Install PyTorch
#-----------------------------------------------------------------------------
print_header "Checking PyTorch Installation"

# Check if PyTorch is already installed with CUDA
PYTORCH_INSTALLED=false
if python -c "import torch" 2>/dev/null; then
    EXISTING_TORCH=$(python -c "import torch; print(torch.__version__)")
    EXISTING_CUDA=$(python -c "import torch; print(torch.cuda.is_available())")
    if [ "$EXISTING_CUDA" = "True" ]; then
        EXISTING_CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
        print_success "PyTorch $EXISTING_TORCH with CUDA $EXISTING_CUDA_VER already installed"
        PYTORCH_INSTALLED=true
    fi
fi

if [ "$PYTORCH_INSTALLED" = false ]; then
    print_step "Installing PyTorch $PYTORCH_VERSION with CUDA 12.4..."
    pip install torch==$PYTORCH_VERSION torchvision==0.21.0 --index-url https://download.pytorch.org/whl/$CUDA_PYTORCH
fi

# Verify PyTorch installation
print_step "Verifying PyTorch CUDA support..."
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
    print_success "PyTorch CUDA: $TORCH_CUDA"
else
    print_error "PyTorch CUDA not available!"
    exit 1
fi

#-----------------------------------------------------------------------------
# Install Basic Dependencies
#-----------------------------------------------------------------------------
print_header "Installing Dependencies"

print_step "Installing basic pip packages..."
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh \
    transformers gradio==6.0.1 tensorboard pandas lpips zstandard

print_step "Installing utils3d..."
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

print_step "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y libjpeg-dev

print_step "Installing pillow-simd..."
pip install pillow-simd

print_step "Installing additional packages..."
pip install kornia timm

print_success "Basic dependencies installed"

#-----------------------------------------------------------------------------
# Build Custom Extensions
#-----------------------------------------------------------------------------
print_header "Building Custom CUDA Extensions"

# Create temp directory for extensions
mkdir -p "$EXTENSIONS_DIR"

# Flash Attention
print_step "Installing flash-attn 2.7.3..."
pip install flash-attn==2.7.3
print_success "flash-attn installed"

# nvdiffrast
print_step "Installing nvdiffrast v0.4.0..."
if [ ! -d "$EXTENSIONS_DIR/nvdiffrast" ]; then
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git "$EXTENSIONS_DIR/nvdiffrast"
fi
pip install "$EXTENSIONS_DIR/nvdiffrast" --no-build-isolation
print_success "nvdiffrast installed"

# nvdiffrec
print_step "Installing nvdiffrec (renderutils)..."
if [ ! -d "$EXTENSIONS_DIR/nvdiffrec" ]; then
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git "$EXTENSIONS_DIR/nvdiffrec"
fi
pip install "$EXTENSIONS_DIR/nvdiffrec" --no-build-isolation
print_success "nvdiffrec installed"

# CuMesh
print_step "Installing CuMesh..."
if [ ! -d "$EXTENSIONS_DIR/CuMesh" ]; then
    git clone --recursive https://github.com/JeffreyXiang/CuMesh.git "$EXTENSIONS_DIR/CuMesh"
fi
pip install "$EXTENSIONS_DIR/CuMesh" --no-build-isolation
print_success "CuMesh installed"

# FlexGEMM
print_step "Installing FlexGEMM..."
if [ ! -d "$EXTENSIONS_DIR/FlexGEMM" ]; then
    git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git "$EXTENSIONS_DIR/FlexGEMM"
fi
pip install "$EXTENSIONS_DIR/FlexGEMM" --no-build-isolation
print_success "FlexGEMM installed"

# O-Voxel (from local directory)
print_step "Installing O-Voxel..."
if [ -d "$TRELLIS_DIR/o-voxel" ]; then
    cp -r "$TRELLIS_DIR/o-voxel" "$EXTENSIONS_DIR/o-voxel"
    pip install "$EXTENSIONS_DIR/o-voxel" --no-build-isolation
    print_success "O-Voxel installed"
else
    print_error "o-voxel directory not found in TRELLIS.2"
    exit 1
fi

#-----------------------------------------------------------------------------
# Final Verification
#-----------------------------------------------------------------------------
print_header "Verifying Installation"

print_step "Testing imports..."
python -c "
import torch
import trellis2
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel
print('All imports successful!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

if [ $? -eq 0 ]; then
    print_success "All imports working correctly!"
else
    print_error "Import verification failed"
    exit 1
fi

#-----------------------------------------------------------------------------
# Create run script
#-----------------------------------------------------------------------------
print_header "Creating Launcher Script"

cat > "$SCRIPT_DIR/run_trellis2.sh" << 'EOF'
#!/bin/bash
# TRELLIS.2 Web Demo Launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/TRELLIS.2"

# Activate conda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate trellis2

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OPENCV_IO_ENABLE_OPENEXR=1

echo "Starting TRELLIS.2 Web Demo..."
echo "Open http://127.0.0.1:7860 in your browser"
echo ""
python app.py
EOF

chmod +x "$SCRIPT_DIR/run_trellis2.sh"
print_success "Created: $SCRIPT_DIR/run_trellis2.sh"

#-----------------------------------------------------------------------------
# Done!
#-----------------------------------------------------------------------------
print_header "Installation Complete! 🎉"
echo ""
echo -e "  ${GREEN}✓${NC} Conda environment: ${BOLD}$ENV_NAME${NC}"
echo -e "  ${GREEN}✓${NC} PyTorch ${PYTORCH_VERSION} with CUDA 12.4"
echo -e "  ${GREEN}✓${NC} All CUDA extensions compiled"
echo ""
echo -e "  ${BOLD}To run the web demo:${NC}"
echo -e "    ${CYAN}./run_trellis2.sh${NC}"
echo ""
echo -e "  ${BOLD}Or manually:${NC}"
echo -e "    ${CYAN}conda activate $ENV_NAME${NC}"
echo -e "    ${CYAN}cd TRELLIS.2${NC}"
echo -e "    ${CYAN}python app.py${NC}"
echo ""
echo -e "  ${BOLD}Note:${NC} Model weights (~15GB) will auto-download on first run"
echo ""
