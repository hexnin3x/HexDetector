# Makefile for HexDetector Docker Management
# Quick commands for building, running, and managing Docker containers

.PHONY: help build up down run-demo run-full run-custom clean logs shell test jupyter

# Default target
.DEFAULT_GOAL := help

# Variables
IMAGE_NAME := hexdetector
VERSION := latest
DOCKER_COMPOSE := docker-compose

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)HexDetector Docker Management$(NC)"
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'

build: ## Build Docker images
	@echo "$(BLUE)Building HexDetector Docker image...$(NC)"
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)✓ Build complete!$(NC)"

up: ## Start all services in background
	@echo "$(BLUE)Starting HexDetector services...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Services started!$(NC)"

down: ## Stop all services
	@echo "$(BLUE)Stopping HexDetector services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Services stopped!$(NC)"

run-demo: ## Run quick demo with XGBoost
	@echo "$(BLUE)Running demo mode...$(NC)"
	$(DOCKER_COMPOSE) run --rm hexdetector python src/main.py --mode demo --model xgboost

run-full: ## Run full analysis with Random Forest
	@echo "$(BLUE)Running full analysis (this may take hours)...$(NC)"
	$(DOCKER_COMPOSE) run --rm hexdetector python src/main.py --mode full --model random_forest

run-custom: ## Run custom analysis (usage: make run-custom SAMPLES=50000 MODEL=svm)
	@echo "$(BLUE)Running custom analysis...$(NC)"
	$(DOCKER_COMPOSE) run --rm hexdetector python src/main.py --mode custom --samples $(or $(SAMPLES),10000) --model $(or $(MODEL),random_forest)

run-rf: ## Run demo with Random Forest
	$(DOCKER_COMPOSE) run --rm hexdetector python src/main.py --mode demo --model random_forest

run-xgb: ## Run demo with XGBoost
	$(DOCKER_COMPOSE) run --rm hexdetector python src/main.py --mode demo --model xgboost

run-svm: ## Run demo with SVM
	$(DOCKER_COMPOSE) run --rm hexdetector python src/main.py --mode demo --model svm

run-nb: ## Run demo with Naive Bayes
	$(DOCKER_COMPOSE) run --rm hexdetector python src/main.py --mode demo --model naive_bayes

logs: ## View container logs
	@echo "$(BLUE)Showing logs (Ctrl+C to exit)...$(NC)"
	$(DOCKER_COMPOSE) logs -f hexdetector

shell: ## Open bash shell in container
	@echo "$(BLUE)Opening shell in HexDetector container...$(NC)"
	$(DOCKER_COMPOSE) run --rm hexdetector /bin/bash

test: ## Run unit tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(DOCKER_COMPOSE) run --rm hexdetector python -m unittest discover tests/ -v

check-config: ## Check configuration
	@echo "$(BLUE)Checking configuration...$(NC)"
	$(DOCKER_COMPOSE) run --rm hexdetector python src/utils/check_config.py

jupyter: ## Start Jupyter Lab
	@echo "$(BLUE)Starting Jupyter Lab...$(NC)"
	@echo "$(GREEN)Access at: http://localhost:8888$(NC)"
	$(DOCKER_COMPOSE) up hexdetector-notebook

clean: ## Remove containers and images
	@echo "$(BLUE)Cleaning up containers and images...$(NC)"
	$(DOCKER_COMPOSE) down -v
	docker rmi $(IMAGE_NAME):$(VERSION) || true
	@echo "$(GREEN)✓ Cleanup complete!$(NC)"

clean-all: ## Remove everything including volumes and cached data
	@echo "$(RED)Warning: This will remove all containers, images, and volumes!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(DOCKER_COMPOSE) down -v; \
		docker rmi $(IMAGE_NAME):$(VERSION) || true; \
		docker system prune -af; \
		echo "$(GREEN)✓ Complete cleanup done!$(NC)"; \
	fi

rebuild: ## Rebuild without cache
	@echo "$(BLUE)Rebuilding without cache...$(NC)"
	$(DOCKER_COMPOSE) build --no-cache
	@echo "$(GREEN)✓ Rebuild complete!$(NC)"

ps: ## Show running containers
	@echo "$(BLUE)Running containers:$(NC)"
	docker ps --filter name=hexdetector

stats: ## Show container resource usage
	@echo "$(BLUE)Container resource usage:$(NC)"
	docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

push: ## Push image to Docker Hub
	@echo "$(BLUE)Pushing to Docker Hub...$(NC)"
	docker tag $(IMAGE_NAME):$(VERSION) hexnin3x/$(IMAGE_NAME):$(VERSION)
	docker push hexnin3x/$(IMAGE_NAME):$(VERSION)
	@echo "$(GREEN)✓ Push complete!$(NC)"

pull: ## Pull image from Docker Hub
	@echo "$(BLUE)Pulling from Docker Hub...$(NC)"
	docker pull hexnin3x/$(IMAGE_NAME):$(VERSION)
	@echo "$(GREEN)✓ Pull complete!$(NC)"

version: ## Show version information
	@echo "$(BLUE)HexDetector Version Information:$(NC)"
	@echo "Image: $(IMAGE_NAME):$(VERSION)"
	@docker --version
	@docker-compose --version
	@$(DOCKER_COMPOSE) run --rm hexdetector python --version

install-deps: ## Install local development dependencies
	@echo "$(BLUE)Installing local dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed!$(NC)"

setup-data: ## Create data directories
	@echo "$(BLUE)Creating data directories...$(NC)"
	mkdir -p data/raw data/processed output logs models
	chmod -R 755 data output logs models
	@echo "$(GREEN)✓ Directories created!$(NC)"

backup: ## Backup output and models
	@echo "$(BLUE)Creating backup...$(NC)"
	tar -czf hexdetector-backup-$$(date +%Y%m%d-%H%M%S).tar.gz output/ models/ logs/
	@echo "$(GREEN)✓ Backup created!$(NC)"

# Development helpers
dev-build: ## Build and run in one command
	@make build
	@make run-demo

dev-test: ## Build, run tests
	@make build
	@make test

dev-shell: ## Build and open shell
	@make build
	@make shell
