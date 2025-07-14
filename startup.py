#!/usr/bin/env python3
"""
Startup script for Railway deployment.
Handles model downloading and app initialization.
"""

import os
import sys
import subprocess
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_environment():
    """Set up the environment for Railway deployment."""
    print("Setting up Drishya Flask App for Railway...")
    
    # Create necessary directories
    temp_dir = os.getenv('TEMP_DIR', '/tmp/drishya_temp')
    model_cache_dir = os.getenv('MODEL_CACHE_DIR', '/tmp/drishya_models')
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)
    
    print(f"Created temp directory: {temp_dir}")
    print(f"Created model cache directory: {model_cache_dir}")
    
    return True

def start_app():
    """Start the Flask application using Gunicorn."""
    port = os.getenv('PORT', '5000')
    workers = os.getenv('WEB_CONCURRENCY', '1')

    print(f"Starting Flask app on port {port} with {workers} workers...")

    # Memory-optimized Gunicorn configuration for 512MB RAM
    cmd = [
        'gunicorn',
        '--bind', f'0.0.0.0:{port}',
        '--workers', '1',  # Force single worker for memory constraints
        '--timeout', '300',  # 5 minutes timeout for model loading
        '--worker-class', 'sync',
        '--max-requests', '100',  # Lower to prevent memory accumulation
        '--max-requests-jitter', '10',
        '--worker-tmp-dir', '/dev/shm',  # Use shared memory for better performance
        '--no-sendfile',  # Reduce memory usage
        # Remove --preload to avoid loading model in master process
        'app:app'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Application stopped by user")
        sys.exit(0)

def main():
    """Main startup function."""
    try:
        # Setup environment
        setup_environment()
        
        # Start the application
        start_app()
        
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
