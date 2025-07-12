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
