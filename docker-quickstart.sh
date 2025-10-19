#!/bin/bash

# HexDetector Docker Quick Start
# This script helps you get started with HexDetector using Docker

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║           🐳 HexDetector Docker Quick Start 🐳                ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed!"
    echo "Please install Docker Compose from: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker is installed: $(docker --version)"
echo "✅ Docker Compose is installed: $(docker-compose --version)"
echo ""

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker daemon is running"
echo ""

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/raw data/processed output logs models
echo "✅ Directories created"
echo ""

# Build Docker image
echo "🔨 Building Docker image (this may take a few minutes)..."
if docker-compose build; then
    echo "✅ Docker image built successfully!"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi
echo ""

# Show menu
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  What would you like to do?                                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "1. Run Quick Demo (XGBoost, ~2 minutes)"
echo "2. Run Quick Demo (Random Forest)"
echo "3. Run All Models Demo"
echo "4. Start Jupyter Lab (http://localhost:8888)"
echo "5. Open Shell in Container"
echo "6. Run Tests"
echo "7. Check Configuration"
echo "8. Show Help"
echo "9. Exit"
echo ""

read -p "Enter your choice [1-9]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Running demo with XGBoost..."
        docker-compose run --rm hexdetector python src/main.py --mode demo --model xgboost
        echo ""
        echo "✅ Demo complete! Check ./output/ for results"
        ;;
    2)
        echo ""
        echo "🚀 Running demo with Random Forest..."
        docker-compose run --rm hexdetector python src/main.py --mode demo --model random_forest
        echo ""
        echo "✅ Demo complete! Check ./output/ for results"
        ;;
    3)
        echo ""
        echo "🚀 Running demo with all models..."
        for model in random_forest xgboost svm naive_bayes logistic decision_tree gradient_boosting; do
            echo "Testing $model..."
            docker-compose run --rm hexdetector python src/main.py --mode demo --model $model
        done
        echo ""
        echo "✅ All models tested! Check ./output/ for results"
        ;;
    4)
        echo ""
        echo "📓 Starting Jupyter Lab..."
        echo "Access at: http://localhost:8888"
        echo "Press Ctrl+C to stop"
        echo ""
        docker-compose up hexdetector-notebook
        ;;
    5)
        echo ""
        echo "🐚 Opening shell in container..."
        docker-compose run --rm hexdetector /bin/bash
        ;;
    6)
        echo ""
        echo "🧪 Running tests..."
        docker-compose run --rm hexdetector python -m unittest discover tests/ -v
        ;;
    7)
        echo ""
        echo "⚙️  Checking configuration..."
        docker-compose run --rm hexdetector python src/utils/check_config.py
        ;;
    8)
        echo ""
        echo "📚 Available commands:"
        echo ""
        echo "Using Make:"
        echo "  make build          - Build Docker images"
        echo "  make run-demo       - Run quick demo"
        echo "  make run-xgb        - Run demo with XGBoost"
        echo "  make run-rf         - Run demo with Random Forest"
        echo "  make jupyter        - Start Jupyter Lab"
        echo "  make shell          - Open shell"
        echo "  make test           - Run tests"
        echo "  make logs           - View logs"
        echo "  make clean          - Clean up"
        echo ""
        echo "Using Docker Compose:"
        echo "  docker-compose run --rm hexdetector python src/main.py --mode demo"
        echo "  docker-compose run --rm hexdetector python src/main.py --mode full"
        echo "  docker-compose up hexdetector-notebook"
        echo ""
        echo "See DOCKER_GUIDE.md for complete documentation"
        ;;
    9)
        echo ""
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo ""
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Next Steps:                                                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "• View results in ./output/ directory"
echo "• Check logs in ./logs/ directory"
echo "• See DOCKER_GUIDE.md for more commands"
echo "• Run 'make help' to see all available commands"
echo ""
echo "🎉 Happy analyzing!"
