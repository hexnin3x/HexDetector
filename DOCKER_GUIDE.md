# 🐳 HexDetector Docker Guide

Complete guide to running HexDetector in Docker containers.

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and run
docker-compose up -d

# Run demo mode
docker-compose run --rm hexdetector python src/main.py --mode demo --model xgboost

# View logs
docker-compose logs -f hexdetector

# Stop
docker-compose down
```

### Option 2: Docker CLI

```bash
# Build image
docker build -t hexdetector:latest .

# Run demo
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output \
  hexdetector:latest python src/main.py --mode demo --model random_forest

# Interactive shell
docker run -it --rm hexdetector:latest /bin/bash
```

---

## 📦 What's Included

### **Base Image**
- Python 3.9 slim (optimized for size)
- All dependencies from `requirements.txt`
- Non-root user for security

### **Pre-configured Volumes**
- `/app/data` - Dataset storage
- `/app/output` - Results and visualizations
- `/app/logs` - Application logs
- `/app/models` - Trained models

### **Services**
1. **hexdetector** - Main ML pipeline
2. **hexdetector-notebook** - Jupyter Lab (optional)

---

## 🔧 Installation & Setup

### Prerequisites

- Docker Engine 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose 1.29+ ([Install Compose](https://docs.docker.com/compose/install/))
- 8GB+ RAM available for Docker
- 10GB+ disk space

### Verify Installation

```bash
docker --version
docker-compose --version
```

---

## 🎯 Usage Examples

### 1. **Quick Demo (10K samples)**

```bash
docker-compose run --rm hexdetector \
  python src/main.py --mode demo --model xgboost
```

### 2. **Full Analysis**

```bash
# Prepare your data first in ./data/raw/
docker-compose run --rm hexdetector \
  python src/main.py --mode full --model random_forest
```

### 3. **Custom Analysis**

```bash
docker-compose run --rm hexdetector \
  python src/main.py --mode custom --samples 50000 --model svm
```

### 4. **Train Multiple Models**

```bash
# Random Forest
docker-compose run --rm hexdetector \
  python src/main.py --mode demo --model random_forest

# XGBoost
docker-compose run --rm hexdetector \
  python src/main.py --mode demo --model xgboost

# SVM
docker-compose run --rm hexdetector \
  python src/main.py --mode demo --model svm
```

### 5. **Check Configuration**

```bash
docker-compose run --rm hexdetector \
  python src/utils/check_config.py
```

### 6. **Run Tests**

```bash
docker-compose run --rm hexdetector \
  python -m unittest discover tests/
```

---

## 📊 Jupyter Notebook (Interactive Analysis)

### Start Jupyter Lab

```bash
docker-compose up hexdetector-notebook
```

### Access Jupyter

Open in browser: http://localhost:8888

No password required (for local development only!)

### Features
- Explore data interactively
- Visualize results
- Test models
- Create custom analyses

---

## 🗂️ Volume Mounting

### Default Volumes (in docker-compose.yml)

```yaml
volumes:
  - ./data:/app/data           # Your datasets
  - ./output:/app/output       # Results
  - ./logs:/app/logs           # Log files
  - ./models:/app/models       # Saved models
```

### Custom Volume Mounting

```bash
# Mount specific dataset directory
docker run --rm \
  -v /path/to/your/iot23:/app/data/raw \
  -v $(pwd)/output:/app/output \
  hexdetector:latest \
  python src/main.py --mode demo
```

---

## ⚙️ Environment Variables

### Available Variables

```bash
# Set in docker-compose.yml or command line
PYTHONUNBUFFERED=1              # Real-time output
IOT23_DATA_DIR=/app/data/raw    # Data directory
IOT23_PROCESSED_DIR=/app/data/processed
LOG_LEVEL=INFO                  # Logging level
```

### Override at Runtime

```bash
docker-compose run --rm \
  -e LOG_LEVEL=DEBUG \
  hexdetector python src/main.py --mode demo
```

---

## 🔨 Building Custom Images

### Build with Custom Tag

```bash
docker build -t hexdetector:v1.0.0 .
```

### Build with Build Args

```bash
docker build \
  --build-arg PYTHON_VERSION=3.10 \
  -t hexdetector:python3.10 .
```

### Multi-platform Build

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t hexdetector:multiarch .
```

---

## 💾 Data Persistence

### Prepare Your Data Directory

```bash
# Create directory structure
mkdir -p data/raw data/processed output logs models

# Copy IoT-23 dataset
cp -r /path/to/iot23/* data/raw/

# Set permissions
chmod -R 755 data/
```

### Backup Results

```bash
# Results are automatically saved to ./output/
tar -czf hexdetector-results-$(date +%Y%m%d).tar.gz output/
```

---

## 🚀 Performance Optimization

### Resource Limits

Edit `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '8'      # Use 8 CPUs
      memory: 16G    # Use 16GB RAM
    reservations:
      cpus: '4'      # Reserve 4 CPUs
      memory: 8G     # Reserve 8GB RAM
```

### Enable GPU Support (NVIDIA)

1. Install [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

2. Add to `docker-compose.yml`:
```yaml
services:
  hexdetector:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

3. Use GPU-enabled models (XGBoost, TensorFlow)

---

## 🐛 Troubleshooting

### Issue: Container Exits Immediately

```bash
# Check logs
docker-compose logs hexdetector

# Run with shell to debug
docker-compose run --rm hexdetector /bin/bash
```

### Issue: Permission Denied

```bash
# Fix file permissions
sudo chown -R $USER:$USER data/ output/ logs/
chmod -R 755 data/ output/ logs/
```

### Issue: Out of Memory

```bash
# Increase Docker memory limit in Docker Desktop settings
# Or use custom mode with smaller sample size
docker-compose run --rm hexdetector \
  python src/main.py --mode custom --samples 10000
```

### Issue: Module Not Found

```bash
# Rebuild image
docker-compose build --no-cache hexdetector

# Verify installation
docker-compose run --rm hexdetector pip list
```

---

## 🔍 Useful Docker Commands

### Container Management

```bash
# List running containers
docker ps

# List all containers
docker ps -a

# Stop container
docker stop hexdetector

# Remove container
docker rm hexdetector

# View container logs
docker logs -f hexdetector
```

### Image Management

```bash
# List images
docker images

# Remove image
docker rmi hexdetector:latest

# Clean up unused images
docker image prune -a

# View image details
docker inspect hexdetector:latest
```

### Cleanup

```bash
# Remove all stopped containers
docker container prune

# Remove all unused images
docker image prune -a

# Remove all unused volumes
docker volume prune

# Complete cleanup
docker system prune -a --volumes
```

---

## 📈 Production Deployment

### Docker Registry

```bash
# Tag for registry
docker tag hexdetector:latest registry.example.com/hexdetector:latest

# Push to registry
docker push registry.example.com/hexdetector:latest

# Pull on production
docker pull registry.example.com/hexdetector:latest
```

### Docker Hub

```bash
# Tag for Docker Hub
docker tag hexdetector:latest hexnin3x/hexdetector:latest

# Login
docker login

# Push
docker push hexnin3x/hexdetector:latest
```

### Run in Production

```bash
# Use specific version
docker run -d \
  --name hexdetector-prod \
  --restart always \
  -v /data/iot23:/app/data/raw \
  -v /data/output:/app/output \
  hexnin3x/hexdetector:v1.0.0 \
  python src/main.py --mode full --model xgboost
```

---

## 🌐 Kubernetes Deployment (Advanced)

### Create Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hexdetector
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hexdetector
  template:
    metadata:
      labels:
        app: hexdetector
    spec:
      containers:
      - name: hexdetector
        image: hexnin3x/hexdetector:latest
        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
          requests:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: output
          mountPath: /app/output
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: hexdetector-data-pvc
      - name: output
        persistentVolumeClaim:
          claimName: hexdetector-output-pvc
```

---

## ✅ Best Practices

1. **Always use volumes** for data persistence
2. **Set resource limits** to prevent memory issues
3. **Use specific tags** instead of `latest` in production
4. **Run as non-root user** (already configured)
5. **Regular backups** of output and models
6. **Monitor logs** for errors
7. **Update base image** regularly for security patches

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [HexDetector Main Documentation](README.md)

---

## 🎯 Quick Reference

```bash
# Build
docker-compose build

# Start services
docker-compose up -d

# Run demo
docker-compose run --rm hexdetector python src/main.py --mode demo

# Start Jupyter
docker-compose up hexdetector-notebook

# View logs
docker-compose logs -f

# Stop all
docker-compose down

# Cleanup
docker system prune -a
```

---

**Made with 🐳 by hexnin3x**

**HexDetector** - Dockerized for Easy Deployment! 🚀
