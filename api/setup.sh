#!/bin/bash
# Quick setup script for T4L NFL API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🚀 T4L NFL User & Preference API Setup"
echo "======================================"
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    print_error "Please run this script from the api/ directory"
    exit 1
fi

# Check if venv exists
if [ ! -d "../venv" ]; then
    print_error "Virtual environment not found. Please create it first:"
    echo "  cd .."
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if .env exists
if [ ! -f "../.env" ]; then
    print_warning ".env file not found. Creating from example..."
    if [ -f "../.env.example" ]; then
        cp ../.env.example ../.env
        print_warning "Please edit ../.env with your actual Supabase credentials"
    else
        print_error "No .env.example found. Please create ../.env manually"
        exit 1
    fi
fi

print_status "Setup complete! Now you can:"
echo ""
echo "1. Activate virtual environment:"
echo "   cd .."
echo "   source venv/bin/activate"
echo ""
echo "2. Run the API:"
echo "   cd api"
echo "   python main.py"
echo ""
echo "3. Access the API at:"
echo "   - http://localhost:8000 (API)"
echo "   - http://localhost:8000/docs (Documentation)"
echo ""
print_success "Ready to go! 🎉"
