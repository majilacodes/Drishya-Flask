# Deploying Drishya on 512MB RAM

This guide provides specific instructions for deploying Drishya Flask app on Railway's 512MB RAM plan.

## Memory Optimizations Applied

The app has been optimized for 512MB deployment with the following changes:

### 1. Model Loading
- **Lazy Loading**: Model loads only when first needed (not at startup)
- **CPU Only**: Forces CPU usage to avoid GPU memory overhead
- **Single Thread**: PyTorch uses only 1 thread to reduce memory

### 2. Image Processing
- **Smaller Images**: Max width reduced to 1280px (from 1920px)
- **Immediate Cleanup**: Memory freed immediately after processing
- **Efficient Resizing**: Uses INTER_AREA for better memory usage

### 3. Server Configuration
- **Single Worker**: Only 1 Gunicorn worker to minimize memory usage
- **Lower Max Requests**: Reduced to 100 requests per worker restart
- **Shared Memory**: Uses /dev/shm for temporary files

## Railway Deployment Steps

### 1. Environment Variables

Set these exact environment variables in Railway:

```
SECRET_KEY=your-super-secret-key-change-this
APP_PASSWORD=your-secure-password
FLASK_ENV=production
PORT=5000
WEB_CONCURRENCY=1
WORKERS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
TEMP_DIR=/tmp/drishya_temp
```

### 2. Expected Memory Usage

- **Startup**: ~80-120MB
- **After Model Load**: ~300-400MB
- **During Processing**: ~450-500MB
- **Peak Usage**: Should stay under 512MB

### 3. Performance Expectations

- **First Request**: 30-60 seconds (model loading)
- **Subsequent Requests**: 3-8 seconds
- **Concurrent Users**: 1-2 users max
- **Image Size Limit**: 1280px width max

### 4. Monitoring

Use the `/health` endpoint to monitor:
- Memory usage
- Model loading status
- System resources

Example health check response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "system": {
    "memory_available_gb": 0.1,
    "memory_total_gb": 0.5
  }
}
```

### 5. Troubleshooting

**If you see "Worker timeout" or "SIGKILL" errors:**

1. Check memory usage via `/health`
2. Reduce image upload size
3. Verify environment variables are set correctly
4. Check Railway logs for memory spikes

**Memory optimization tips:**
- Upload smaller images (< 2MB recommended)
- Avoid multiple concurrent requests
- Let the app idle between heavy processing

## Success Indicators

✅ App starts without immediate crashes
✅ `/health` shows memory_available_gb > 0.05
✅ First model load completes successfully
✅ Image processing works without timeouts
✅ Memory usage stays under 500MB during processing

## Limitations on 512MB

- Single concurrent user recommended
- Larger images (>5MB) may cause issues
- Processing time slightly slower due to optimizations
- No GPU acceleration available
