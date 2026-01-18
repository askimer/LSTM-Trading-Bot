# RL Algorithmic Trading Bot - Docker Installation Guide

This guide explains how to run the RL Algorithmic Trading Bot using Docker containers.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system

## Setup Instructions

### 1. Build and Run the Container

```bash
# Build and start the container (this will take some time for the first build)
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

### 2. Alternative: Build and Run Separately

```bash
# Build the image
docker build -t rl-trading-bot .

# Run the container interactively
docker run -it --rm \
  -v $(pwd):/app \
  -v $(pwd)/btc_usdt_data:/app/btc_usdt_data \
  -v $(pwd)/btc_usdt_training_data:/app/btc_usdt_training_data \
  -v $(pwd)/rl_logs:/app/rl_logs \
  -v $(pwd)/rl_tensorboard:/app/rl_tensorboard \
  rl-trading-bot

# Or run with custom parameters
docker run -it --rm \
  -v $(pwd):/app \
  rl-trading-bot python train_rl.py --timesteps 100000
```

### 3. Access TensorBoard

To visualize training metrics:

```bash
# Start tensorboard service
docker-compose up tensorboard

# Then access TensorBoard at http://localhost:6006
```

## Docker Services

The docker-compose.yml defines two services:

1. **rl-trading-bot**: Main service running the RL training
   - Maps project directory to `/app`
   - Mounts data and log directories
   - Default command runs `train_rl.py` with 50,000 timesteps

2. **tensorboard**: Service for visualizing training metrics
   - Exposes port 6006
   - Mounts rl_tensorboard directory for logs

## Useful Commands

```bash
# View container logs
docker-compose logs rl-trading-bot

# Execute commands inside the running container
docker-compose exec rl-trading-bot bash

# Stop all services
docker-compose down

# Remove containers, networks, and images
docker-compose down --rmi all

# Run tests inside the container
docker-compose exec rl-trading-bot python -m pytest tests/
```

## Troubleshooting

1. **Permission Issues**: If you encounter permission issues with mounted volumes, ensure your user has appropriate permissions to the project directory.

2. **Memory Issues**: RL training can be memory-intensive. If you experience OOM errors, consider reducing the number of parallel environments with `--n_envs` parameter.

3. **Build Failures**: If the build fails due to TA-Lib installation, ensure that the system has enough resources during compilation.

## Notes

- The Docker container includes all necessary dependencies to run the RL trading bot
- All data and logs are persisted in mounted volumes
- The container uses CPU version of PyTorch for wider compatibility
- For GPU acceleration, you would need to modify the Dockerfile to install CUDA-enabled PyTorch
