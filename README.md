# Drishya - Flask Version

AI-powered product replacement tool using MobileSAM (lightweight Segment Anything Model), converted from Streamlit to Flask.

## Features

- **AI-Powered Segmentation**: Uses MobileSAM model for fast and precise object segmentation
- **Interactive Drawing**: Draw bounding boxes to select products for replacement
- **Advanced Blending**: Feathered edges and color grading for realistic results
- **Real-time Preview**: See results before downloading
- **Password Protection**: Secure access with password authentication
- **Lightweight Model**: Uses pre-loaded MobileSAM weights for fast processing without downloads

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and go to `http://localhost:5000`

## Usage

1. **Login**: Enter the password "setuftw" to access the application
2. **Upload Image**: Upload an image containing a product to replace
3. **Draw Bounding Box**: Click and drag to draw a rectangle around the product
4. **Generate Mask**: Click "Generate Mask" to create the segmentation mask
5. **Upload Product**: Upload the new product image
6. **Adjust Settings**: Fine-tune scale, blending, and color grading options
7. **Replace Product**: Click "Replace Product" to generate the final result
8. **Download**: Download the final image with the replaced product

## Settings

- **Product Scale**: Control the size of the replacement product (0.5x to 2.0x)
- **Edge Blending**: Enable/disable advanced edge blending for smoother transitions
- **Edge Feathering**: Control the softness of edges (0-30 pixels)
- **Color Grading**: Match product colors to the target area or entire image
- **Color Grade Strength**: Control how strongly to apply color matching (0-100%)

## Technical Details

- **Backend**: Flask web framework
- **AI Model**: MobileSAM with ViT-T backbone (lightweight and fast)
- **Image Processing**: OpenCV for advanced blending and color grading
- **Frontend**: Vanilla JavaScript with HTML5 Canvas for interactive drawing

## Password

Default password: `setuftw`

## Model Information

The MobileSAM model weights are included in the repository:
- **MobileSAM**: ~40MB (pre-loaded, no download required)
- **Processing**: Fast inference with minimal memory usage

## Browser Compatibility

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Model Performance

MobileSAM provides excellent performance characteristics:

- **Accuracy**: Very good segmentation quality (comparable to full SAM)
- **Model Size**: ~40MB (10x smaller than regular SAM)
- **Processing Time**: 1-5 seconds per image (up to 5x faster)
- **Memory Usage**: Much lower than regular SAM
- **Best for**: Fast processing, edge computing, production deployments

## Performance

- GPU acceleration supported (CUDA)
- Fallback to CPU if GPU unavailable
- Model loading time: Instant (pre-loaded weights)
- Processing time: 1-5 seconds per image
- Memory efficient: Optimized for production use

## Railway Deployment

This app is ready for deployment on Railway. Follow these steps:

### 1. Prepare for Deployment

The app includes all necessary configuration files:
- `railway.json` - Railway deployment configuration
- `Procfile` - Process definition for Railway
- `startup.py` - Production startup script with Gunicorn
- `.env.example` - Environment variables template

### 2. Deploy to Railway

1. **Connect Repository**:
   - Go to [Railway](https://railway.app)
   - Create a new project from your GitHub repository

2. **Set Environment Variables**:
   ```
   SECRET_KEY=your-super-secret-key-here
   APP_PASSWORD=your-secure-password
   FLASK_ENV=production
   PORT=5000
   MODEL_CACHE_DIR=/tmp/drishya_models
   TEMP_DIR=/tmp/drishya_temp
   ```

3. **Deploy**:
   - Railway will automatically detect the configuration
   - The app will build and deploy using the startup script
   - Model weights will be downloaded on first startup (may take 2-3 minutes)

### 3. Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `SECRET_KEY` | Flask secret key for sessions | `your-secret-key-change-this` | ✅ |
| `APP_PASSWORD` | Application access password | `setuftw` | ✅ |
| `FLASK_ENV` | Flask environment | `development` | ✅ |
| `PORT` | Port for the application | `5000` | ✅ |
| `TEMP_DIR` | Directory for temporary files | `/tmp/drishya_temp` | ❌ |

### 4. Health Check

The app includes a comprehensive health check endpoint at `/health` that provides:
- System information (CPU, memory, disk usage)
- Model loading status
- Directory accessibility
- Environment details

### 5. Production Features

- **Gunicorn WSGI Server**: Production-ready server with proper worker management
- **Environment Configuration**: All settings configurable via environment variables
- **Automatic Model Download**: SAM model weights downloaded and cached automatically
- **Health Monitoring**: Detailed health check for Railway monitoring
- **Error Handling**: Robust error handling for production deployment
- **Resource Management**: Optimized for Railway's resource constraints

### 6. Memory-Optimized Deployment (512MB RAM)

For deployment on Railway's 512MB plan, use these optimized settings:

**Environment Variables for 512MB:**
```
SECRET_KEY=your-super-secret-key-here
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

**Memory Optimizations Applied:**
- Single worker configuration
- Lazy model loading (loads on first use)
- Aggressive image resizing (max 1280px width)
- Immediate memory cleanup after processing
- CPU-only PyTorch with minimal threads
- Reduced max requests per worker

### 7. Troubleshooting

- **Model Loading Issues**: Check logs for download progress and errors
- **Memory Issues**: App optimized for 512MB RAM with MobileSAM
- **Timeout Issues**: Initial model load may take 30-60 seconds on first use
- **Health Check**: Use `/health` endpoint to diagnose issues and monitor memory usage
