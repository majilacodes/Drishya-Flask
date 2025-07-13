# Drishya - Flask Version

AI-powered product replacement tool using Meta's Segment Anything Model (SAM), converted from Streamlit to Flask.

## Features

- **AI-Powered Segmentation**: Uses Meta's SAM model for precise object segmentation
- **Interactive Drawing**: Draw bounding boxes to select products for replacement
- **Advanced Blending**: Feathered edges and color grading for realistic results
- **Real-time Preview**: See results before downloading
- **Password Protection**: Secure access with password authentication

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
- **AI Model**: Meta's Segment Anything Model (SAM) with ViT-B backbone
- **Image Processing**: OpenCV for advanced blending and color grading
- **Frontend**: Vanilla JavaScript with HTML5 Canvas for interactive drawing

## Password

Default password: `setuftw`

## Model Download

The SAM model weights (~375MB) will be automatically downloaded on first use and cached in `~/.cache/drishya/models/`.

## Browser Compatibility

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Performance

- GPU acceleration supported (CUDA)
- Fallback to CPU if GPU unavailable
- Model loading time: 2-3 minutes on first run
- Processing time: 5-15 seconds per image

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
| `MODEL_CACHE_DIR` | Directory for model storage | `/tmp/drishya_models` | ❌ |
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

### 6. Troubleshooting

- **Model Loading Issues**: Check logs for download progress and errors
- **Memory Issues**: Model requires ~2GB RAM, ensure adequate Railway plan
- **Timeout Issues**: Initial startup may take 2-3 minutes for model download
- **Health Check**: Use `/health` endpoint to diagnose issues
