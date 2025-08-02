#!/bin/bash
# Docker helper script for T4L NFL User & Preference API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
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

# Function to check if .env file exists
check_env_file() {
    if [ ! -f "../.env" ]; then
        print_warning ".env file not found in parent directory"
        print_warning "Make sure to set SUPABASE_URL and SUPABASE_KEY environment variables"
        print_warning "You can create a .env file with:"
        echo "SUPABASE_URL=your_supabase_url"
        echo "SUPABASE_KEY=your_supabase_key"
        echo "DEEPSEEK_API_KEY=your_deepseek_key  # Optional"
        echo "LOG_LEVEL=INFO"
    fi
}

# Function to build the Docker image
build_image() {
    print_status "Building Docker image..."
    docker build -f Dockerfile -t t4l-nfl-api:latest ..
    print_success "Docker image built successfully!"
}

# Function to run the container
run_container() {
    print_status "Starting container..."
    
    # Check if container is already running
    if [ "$(docker ps -q -f name=t4l-api)" ]; then
        print_warning "Container 't4l-api' is already running. Stopping it first..."
        docker stop t4l-api
        docker rm t4l-api
    fi
    
    # Run the container
    docker run -d \
        --name t4l-api \
        --env-file ../.env \
        -p 8000:8000 \
        t4l-nfl-api:latest
        
    print_success "Container started successfully!"
    print_status "API available at: http://localhost:8000"
    print_status "API documentation: http://localhost:8000/docs"
    print_status "Health check: http://localhost:8000/health"
}

# Function to use docker-compose
run_compose() {
    print_status "Starting with docker-compose..."
    check_env_file
    docker-compose up -d --build
    print_success "Services started with docker-compose!"
    print_status "API available at: http://localhost:8000"
    print_status "API documentation: http://localhost:8000/docs"
}

# Function to stop containers
stop_containers() {
    print_status "Stopping containers..."
    
    # Stop docker-compose services
    if [ -f "docker-compose.yml" ]; then
        docker-compose down
    fi
    
    # Stop standalone container
    if [ "$(docker ps -q -f name=t4l-api)" ]; then
        docker stop t4l-api
        docker rm t4l-api
    fi
    
    print_success "Containers stopped!"
}

# Function to view logs
view_logs() {
    if [ "$(docker ps -q -f name=t4l-api)" ]; then
        print_status "Showing logs for t4l-api container..."
        docker logs -f t4l-api
    elif [ "$(docker-compose ps -q)" ]; then
        print_status "Showing logs for docker-compose services..."
        docker-compose logs -f
    else
        print_error "No running containers found!"
    fi
}

# Function to run tests in container
run_tests() {
    print_status "Running tests in container..."
    docker run --rm \
        --env-file ../.env \
        t4l-nfl-api:latest \
        python -m pytest tests/test_fastapi_basic.py tests/test_user_preference_api.py -v
}

# Function to check container health
check_health() {
    print_status "Checking API health..."
    
    # Wait a moment for container to start
    sleep 2
    
    if curl -f -s http://localhost:8000/health > /dev/null; then
        print_success "API is healthy!"
        curl -s http://localhost:8000/health | python -m json.tool
    else
        print_error "API health check failed!"
        print_status "Container logs:"
        view_logs
        exit 1
    fi
}

# Function to show local development instructions
show_local_instructions() {
    print_status "Local Development Instructions"
    echo ""
    echo "For local development, follow these steps:"
    echo ""
    echo "1. Activate virtual environment (REQUIRED):"
    echo "   cd /Users/tobiaslatta/Projects/github/bigsliktobi/T4LAPIs"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Run the API:"
    echo "   cd api"
    echo "   python main.py"
    echo ""
    echo "3. Access the API:"
    echo "   - API: http://localhost:8000"
    echo "   - Docs: http://localhost:8000/docs"
    echo "   - Health: http://localhost:8000/health"
    echo ""
    print_warning "Note: Virtual environment must be activated first!"
}

# Main script logic
case "$1" in
    build)
        build_image
        ;;
    run)
        check_env_file
        build_image
        run_container
        sleep 3
        check_health
        ;;
    compose)
        run_compose
        sleep 3
        check_health
        ;;
    stop)
        stop_containers
        ;;
    logs)
        view_logs
        ;;
    test)
        build_image
        run_tests
        ;;
    health)
        check_health
        ;;
    local)
        show_local_instructions
        ;;
    *)
        echo "Usage: $0 {build|run|compose|stop|logs|test|health|local}"
        echo ""
        echo "Commands:"
        echo "  build     - Build the Docker image"
        echo "  run       - Build and run the container"
        echo "  compose   - Start with docker-compose"
        echo "  stop      - Stop all containers"
        echo "  logs      - View container logs"
        echo "  test      - Run tests in container"
        echo "  health    - Check API health"
        echo "  local     - Instructions for local development"
        echo ""
        echo "Examples:"
        echo "  $0 run      # Build and start the API"
        echo "  $0 compose  # Start with docker-compose"
        echo "  $0 logs     # View logs"
        echo "  $0 stop     # Stop containers"
        echo "  $0 local    # Show local dev instructions"
        exit 1
        ;;
esac
