from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import torch
import numpy as np
import cv2
import os
import requests
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from io import BytesIO
import base64
import json
import warnings
from werkzeug.utils import secure_filename
import tempfile
import uuid
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress the torch.classes warning
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create temp directory for storing images
TEMP_DIR = os.getenv('TEMP_DIR', os.path.join(tempfile.gettempdir(), 'drishya_temp'))
os.makedirs(TEMP_DIR, exist_ok=True)

# Global variables for model
mask_predictor = None
device = None
model_loaded = False

def load_model():
    """Load the SAM model by downloading weights automatically"""
    global mask_predictor, device, model_loaded
    
    # Check if CUDA is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model configuration
    model_type = "vit_b"
    model_filename = "sam_vit_b_01ec64.pth"
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

    # Use a persistent directory for model storage
    model_cache_dir = os.getenv('MODEL_CACHE_DIR')
    if model_cache_dir:
        model_dir = os.path.join(model_cache_dir, "models")
    else:
        home_dir = os.path.expanduser("~")
        model_dir = os.path.join(home_dir, ".cache", "drishya", "models")
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, model_filename)

    # Download the model if it doesn't exist
    if not os.path.exists(checkpoint_path):
        try:
            print("🔄 Downloading SAM model weights... This may take 2-3 minutes on first run.")
            
            # Download with improved error handling
            response = requests.get(model_url, stream=True, timeout=120)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192
            
            # Use a temporary file with proper cleanup
            temp_path = checkpoint_path + ".tmp"

            try:
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Update progress with better formatting
                            if total_size > 0:
                                progress = min(downloaded / total_size, 1.0)
                                mb_downloaded = downloaded / (1024*1024)
                                mb_total = total_size / (1024*1024)
                                print(f"Downloaded {mb_downloaded:.1f} MB / {mb_total:.1f} MB ({progress*100:.1f}%)")

                # Verify file size before moving
                if total_size > 0 and downloaded < total_size * 0.95:  # Allow 5% tolerance
                    raise Exception(f"Download incomplete: {downloaded}/{total_size} bytes")
                
                # Move temp file to final location
                os.rename(temp_path, checkpoint_path)
                print("Download complete! Loading model...")
                
            except Exception as e:
                # Clean up temp file on any error
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                raise e

        except requests.exceptions.Timeout:
            raise Exception("Download timeout. Please check your internet connection and try again.")
        except requests.exceptions.ConnectionError:
            raise Exception("Connection error. Please check your internet connection and try again.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")

    # Verify file exists and has reasonable size before loading
    if not os.path.exists(checkpoint_path):
        raise Exception("Model file not found after download.")
    
    file_size = os.path.getsize(checkpoint_path)
    expected_min_size = 300 * 1024 * 1024  # 300MB minimum
    if file_size < expected_min_size:
        raise Exception(f"Model file appears corrupted (size: {file_size/(1024*1024):.1f}MB). Please refresh to re-download.")

    try:
        # Load the model with better error handling
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        mask_predictor = SamPredictor(sam)
        
        # Test model loading
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask_predictor.set_image(test_image)
        
        model_loaded = True
        return mask_predictor, device

    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

def check_password(password):
    """Check if the provided password is correct"""
    app_password = os.getenv('APP_PASSWORD', 'setuftw')
    return password == app_password

def get_session_id():
    """Get or create a session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def save_temp_image(image_data, suffix=''):
    """Save image data to temporary file and return filename"""
    session_id = get_session_id()
    filename = f"{session_id}_{suffix}_{int(time.time())}.png"
    filepath = os.path.join(TEMP_DIR, filename)

    with open(filepath, 'wb') as f:
        f.write(image_data)

    return filename

def get_temp_image_path(filename):
    """Get full path for temporary image file"""
    return os.path.join(TEMP_DIR, filename)

def cleanup_old_files():
    """Clean up old temporary files (older than 1 hour)"""
    try:
        current_time = time.time()
        for filename in os.listdir(TEMP_DIR):
            filepath = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getctime(filepath)
                if file_age > 3600:  # 1 hour
                    os.remove(filepath)
    except Exception as e:
        print(f"Error cleaning up files: {e}")

def show_mask(mask, image):
    """Apply mask on image for visualization."""
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Create a visualization with mask overlay
    result = image.copy()
    mask_rgb = (mask_image[:,:,:3] * 255).astype(np.uint8)
    mask_alpha = (mask_image[:,:,3:] * 255).astype(np.uint8)

    # Blend where mask exists
    alpha_channel = mask_alpha / 255.0
    for c in range(3):
        result[:,:,c] = result[:,:,c] * (1 - alpha_channel[:,:,0]) + mask_rgb[:,:,c] * alpha_channel[:,:,0]

    return result

def create_feathered_mask(mask, feather_amount=10):
    """Create a feathered mask with smooth edges for better blending."""
    # Ensure mask is binary
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    binary_mask = (mask > 128).astype(np.uint8)

    # Create distance transform from the mask edges
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)

    # Normalize the distance transform
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    # Create inverse distance transform for outside the mask
    inv_binary_mask = 1 - binary_mask
    inv_dist_transform = cv2.distanceTransform(inv_binary_mask, cv2.DIST_L2, 3)
    cv2.normalize(inv_dist_transform, inv_dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    # Create a feathered mask by combining both distance transforms
    feathered_mask = np.ones_like(dist_transform, dtype=np.float32)

    # Apply feathering at the boundaries
    feathered_mask = np.where(
        dist_transform > 0,
        np.minimum(1.0, dist_transform * (feather_amount / 2)),
        feathered_mask
    )

    feathered_mask = np.where(
        inv_dist_transform < feather_amount,
        np.maximum(0.0, 1.0 - (inv_dist_transform / feather_amount)),
        feathered_mask * binary_mask
    )

    return feathered_mask

def apply_color_grading(product_image, target_image, mask, strength=0.5):
    """Apply color grading to make the product match the color tone of the target area."""
    # Ensure mask is binary and has correct shape
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    binary_mask = (mask > 128).astype(np.uint8)

    # Get the region of interest from the target image based on the mask
    y_indices, x_indices = np.where(binary_mask == 1)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return product_image  # No adjustment if mask is empty

    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # Extract the target region
    target_region = target_image[y_min:y_max+1, x_min:x_max+1]

    # Create a mask for the target region
    local_mask = binary_mask[y_min:y_max+1, x_min:x_max+1]

    # Apply mask to target region to only consider pixels within the mask
    masked_target = target_region.copy()
    for c in range(3):  # Process each color channel
        masked_target[:,:,c] = masked_target[:,:,c] * local_mask

    # Calculate mean color of the masked target region
    target_pixels = local_mask.sum()
    if target_pixels == 0:
        return product_image  # No pixels to match

    target_means = [
        (masked_target[:,:,c].sum() / target_pixels) for c in range(3)
    ]

    # Calculate standard deviation of the target region for each channel
    target_std = [0, 0, 0]
    for c in range(3):
        # Calculate squared differences for non-zero mask pixels
        squared_diffs = np.zeros_like(local_mask, dtype=float)
        squared_diffs[local_mask > 0] = ((masked_target[:,:,c][local_mask > 0] - target_means[c]) ** 2)
        target_std[c] = np.sqrt(squared_diffs.sum() / target_pixels)

    # Handle alpha channel if present
    has_alpha = product_image.shape[2] == 4
    if has_alpha:
        alpha_channel = product_image[:,:,3].copy()
        product_rgb = product_image[:,:,:3].copy()
    else:
        product_rgb = product_image.copy()

    # Calculate mean and std of the product image (only for non-transparent pixels if has alpha)
    if has_alpha:
        # Only consider pixels that aren't fully transparent
        prod_mask = (alpha_channel > 0).astype(float)
        prod_pixels = prod_mask.sum()
        if prod_pixels == 0:
            return product_image  # No pixels to adjust

        product_means = [
            (product_rgb[:,:,c] * prod_mask).sum() / prod_pixels for c in range(3)
        ]

        product_std = [0, 0, 0]
        for c in range(3):
            squared_diffs = np.zeros_like(prod_mask, dtype=float)
            valid_pixels = prod_mask > 0
            if valid_pixels.sum() > 0:
                squared_diffs[valid_pixels] = ((product_rgb[:,:,c][valid_pixels] - product_means[c]) ** 2)
                product_std[c] = np.sqrt(squared_diffs.sum() / prod_pixels)
    else:
        h, w = product_rgb.shape[:2]
        prod_pixels = h * w
        product_means = [
            product_rgb[:,:,c].sum() / prod_pixels for c in range(3)
        ]

        product_std = [0, 0, 0]
        for c in range(3):
            squared_diffs = (product_rgb[:,:,c] - product_means[c]) ** 2
            product_std[c] = np.sqrt(squared_diffs.sum() / prod_pixels)

    # Perform color grading by adjusting mean and standard deviation
    graded_product = product_rgb.copy().astype(float)

    for c in range(3):
        # Skip channels with zero std to avoid division by zero
        if product_std[c] == 0:
            continue

        # Normalize the product channel
        normalized = (graded_product[:,:,c] - product_means[c]) / product_std[c]

        # Apply target statistics with the specified strength
        if strength < 1.0:
            # Blend between original and target values
            adj_std = product_std[c] * (1 - strength) + target_std[c] * strength
            adj_mean = product_means[c] * (1 - strength) + target_means[c] * strength
        else:
            adj_std = target_std[c]
            adj_mean = target_means[c]

        # Apply the adjustment
        graded_product[:,:,c] = normalized * adj_std + adj_mean

    # Clip values to valid range
    graded_product = np.clip(graded_product, 0, 255).astype(np.uint8)

    # Reattach alpha channel if needed
    if has_alpha:
        graded_product_with_alpha = np.zeros((graded_product.shape[0], graded_product.shape[1], 4), dtype=np.uint8)
        graded_product_with_alpha[:,:,:3] = graded_product
        graded_product_with_alpha[:,:,3] = alpha_channel
        return graded_product_with_alpha

    return graded_product

def replace_product_in_image(ad_image, new_product, mask, scale_factor=1.0, feather_amount=15, use_blending=True):
    """
    Replace a product in an ad image with improved edge blending and error handling.
    """
    try:
        # Input validation
        if ad_image is None or new_product is None or mask is None:
            raise ValueError("Invalid input: image, product, or mask is None")

        if len(ad_image.shape) != 3 or len(new_product.shape) not in [3, 4]:
            raise ValueError("Invalid image dimensions")

        # Ensure mask is binary
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        binary_mask = (mask > 128).astype(np.uint8)

        # Get bounding box from mask
        y_indices, x_indices = np.where(binary_mask == 1)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return ad_image

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Validate bounding box
        if y_max <= y_min or x_max <= x_min:
            return ad_image

        # Calculate dimensions with bounds checking
        mask_height = min(y_max - y_min + 1, ad_image.shape[0] - y_min)
        mask_width = min(x_max - x_min + 1, ad_image.shape[1] - x_min)

        if mask_height <= 0 or mask_width <= 0:
            return ad_image

        # Create output image
        output = ad_image.copy()

        # Get product dimensions and aspect ratio with validation
        prod_height, prod_width = new_product.shape[:2]
        if prod_height <= 0 or prod_width <= 0:
            return ad_image

        prod_aspect_ratio = prod_width / prod_height

        # Calculate the dimensions to preserve aspect ratio
        base_dimension = min(mask_width, mask_height)
        scaled_dimension = max(1, base_dimension * scale_factor)

        print(f"🔧 Mask dimensions: {mask_width}x{mask_height}")
        print(f"🔧 Base dimension: {base_dimension}")
        print(f"🔧 Scale factor: {scale_factor}")
        print(f"🔧 Scaled dimension: {scaled_dimension}")

        # Calculate new dimensions based on aspect ratio
        if prod_aspect_ratio > 1.0:
            resize_width = scaled_dimension * prod_aspect_ratio
            resize_height = scaled_dimension
        else:
            resize_width = scaled_dimension
            resize_height = scaled_dimension / prod_aspect_ratio

        # Ensure minimum dimensions
        resize_width = max(1, int(resize_width))
        resize_height = max(1, int(resize_height))

        print(f"🔧 Final resize dimensions: {resize_width}x{resize_height}")

        # Calculate centering offsets
        offset_x = int((mask_width - resize_width) / 2)
        offset_y = int((mask_height - resize_height) / 2)

        # Create a feathered mask for better edge blending
        mask_roi = binary_mask[y_min:y_max+1, x_min:x_max+1].astype(np.float32)
        feathered_mask = mask_roi
        if use_blending:
            feathered_mask = create_feathered_mask(mask_roi, feather_amount)

        # Handle transparent images (RGBA) vs RGB
        if new_product.shape[2] == 4:
            # RGBA handling with improved error checking
            alpha = new_product[:, :, 3] / 255.0
            rgb = new_product[:, :, :3]

            # Resize with proper interpolation
            resized_rgb = cv2.resize(rgb, (resize_width, resize_height), interpolation=cv2.INTER_LANCZOS4)
            resized_alpha = cv2.resize(alpha, (resize_width, resize_height), interpolation=cv2.INTER_LANCZOS4)

            # Create product mask with bounds checking
            product_mask = np.zeros((mask_height, mask_width), dtype=np.float32)

            # Calculate safe paste coordinates
            paste_y_start = max(0, offset_y)
            paste_y_end = min(mask_height, offset_y + resize_height)
            paste_x_start = max(0, offset_x)
            paste_x_end = min(mask_width, offset_x + resize_width)

            # Calculate corresponding product coordinates
            prod_y_start = max(0, -offset_y)
            prod_y_end = min(resize_height, prod_y_start + (paste_y_end - paste_y_start))
            prod_x_start = max(0, -offset_x)
            prod_x_end = min(resize_width, prod_x_start + (paste_x_end - paste_x_start))

            # Place the resized alpha with bounds checking
            if (paste_y_end > paste_y_start and paste_x_end > paste_x_start and
                prod_y_end > prod_y_start and prod_x_end > prod_x_start):

                try:
                    product_mask[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = \
                        resized_alpha[prod_y_start:prod_y_end, prod_x_start:prod_x_end]
                except ValueError:
                    # Fallback: center the product
                    center_y = mask_height // 2
                    center_x = mask_width // 2
                    half_h = resize_height // 2
                    half_w = resize_width // 2

                    y_start = max(0, center_y - half_h)
                    y_end = min(mask_height, center_y + half_h)
                    x_start = max(0, center_x - half_w)
                    x_end = min(mask_width, center_x + half_w)

                    product_mask[y_start:y_end, x_start:x_end] = 1.0

            # Apply feathered mask
            product_mask = product_mask * feathered_mask
            product_mask_3ch = np.stack([product_mask, product_mask, product_mask], axis=2)

            # Get ROI and create RGB blend image
            roi = output[y_min:y_max+1, x_min:x_max+1]
            rgb_to_blend = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)

            if (paste_y_end > paste_y_start and paste_x_end > paste_x_start and
                prod_y_end > prod_y_start and prod_x_end > prod_x_start):
                try:
                    rgb_to_blend[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = \
                        resized_rgb[prod_y_start:prod_y_end, prod_x_start:prod_x_end]
                except ValueError:
                    pass  # Skip if dimensions don't match

            # Perform blending
            blended_roi = roi * (1 - product_mask_3ch) + rgb_to_blend * product_mask_3ch

        else:
            # RGB handling (similar improvements)
            resized_product = cv2.resize(new_product, (resize_width, resize_height), interpolation=cv2.INTER_LANCZOS4)

            product_to_blend = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)

            # Safe coordinate calculation (same as above)
            paste_y_start = max(0, offset_y)
            paste_y_end = min(mask_height, offset_y + resize_height)
            paste_x_start = max(0, offset_x)
            paste_x_end = min(mask_width, offset_x + resize_width)

            prod_y_start = max(0, -offset_y)
            prod_y_end = min(resize_height, prod_y_start + (paste_y_end - paste_y_start))
            prod_x_start = max(0, -offset_x)
            prod_x_end = min(resize_width, prod_x_start + (paste_x_end - paste_x_start))

            if (paste_y_end > paste_y_start and paste_x_end > paste_x_start and
                prod_y_end > prod_y_start and prod_x_end > prod_x_start):
                try:
                    product_to_blend[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = \
                        resized_product[prod_y_start:prod_y_end, prod_x_start:prod_x_end]
                except ValueError:
                    pass

            # Create product mask
            product_mask = np.zeros((mask_height, mask_width), dtype=np.float32)
            if (paste_y_end > paste_y_start and paste_x_end > paste_x_start):
                product_mask[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = 1

            product_mask = product_mask * feathered_mask
            product_mask_3ch = np.stack([product_mask, product_mask, product_mask], axis=2)

            roi = output[y_min:y_max+1, x_min:x_max+1]
            blended_roi = roi * (1 - product_mask_3ch) + product_to_blend * product_mask_3ch

        # Apply edge blending if enabled
        if use_blending:
            try:
                edge_kernel = np.ones((5, 5), np.uint8)
                edge_mask = cv2.dilate(mask_roi.astype(np.uint8), edge_kernel) - mask_roi.astype(np.uint8)
                edge_mask = np.clip(edge_mask, 0, 1).astype(np.float32)

                # Apply guided filtering with fallback to Gaussian blur
                try:
                    r = 5
                    eps = 0.1
                    harmonized_blend = blended_roi.copy()
                    for c in range(3):
                        harmonized_blend[:,:,c] = cv2.guidedFilter(
                            roi[:,:,c].astype(np.float32),
                            blended_roi[:,:,c].astype(np.float32),
                            r, eps
                        )
                    blended_roi = harmonized_blend.astype(np.uint8)
                except:
                    # Fallback to Gaussian blur
                    blur_amount = 3
                    edge_blur = cv2.GaussianBlur(edge_mask, (blur_amount*2+1, blur_amount*2+1), 0) * 0.7
                    edge_blur_3ch = np.stack([edge_blur, edge_blur, edge_blur], axis=2)
                    harmonized_blend = blended_roi * (1 - edge_blur_3ch) + cv2.GaussianBlur(blended_roi, (blur_amount*2+1, blur_amount*2+1), 0) * edge_blur_3ch
                    blended_roi = harmonized_blend.astype(np.uint8)
            except:
                pass  # Use basic blending if edge processing fails

        # Place the blended region back with bounds checking
        try:
            output[y_min:y_max+1, x_min:x_max+1] = blended_roi
        except ValueError:
            # Fallback: try to place what fits
            out_h, out_w = output.shape[:2]
            roi_h, roi_w = blended_roi.shape[:2]

            actual_y_end = min(y_min + roi_h, out_h)
            actual_x_end = min(x_min + roi_w, out_w)
            actual_roi_h = actual_y_end - y_min
            actual_roi_w = actual_x_end - x_min

            if actual_roi_h > 0 and actual_roi_w > 0:
                output[y_min:actual_y_end, x_min:actual_x_end] = blended_roi[:actual_roi_h, :actual_roi_w]

        return output

    except Exception as e:
        print(f"Error during image processing: {str(e)}")
        return ad_image  # Return original image on error

@app.route('/')
def index():
    """Main page with password protection"""
    if 'authenticated' not in session:
        return render_template('login.html')
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    """Handle login"""
    password = request.form.get('password')
    if check_password(password):
        session['authenticated'] = True
        return redirect(url_for('index'))
    else:
        return render_template('login.html', error="Password incorrect")

@app.route('/logout')
def logout():
    """Handle logout"""
    session.pop('authenticated', None)
    return redirect(url_for('index'))

@app.route('/health')
def health_check():
    """Enhanced health check endpoint for Railway monitoring"""
    import psutil
    import platform

    try:
        # Basic health status
        health_status = {
            "status": "healthy",
            "service": "drishya",
            "model": "SAM",
            "version": "1.0.0",
            "timestamp": time.time(),
            "model_loaded": model_loaded,
            "environment": os.getenv('RAILWAY_ENVIRONMENT', 'development')
        }

        # System information
        health_status["system"] = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "disk_usage_percent": psutil.disk_usage('/').percent
        }

        # Check critical directories
        health_status["directories"] = {
            "temp_dir_exists": os.path.exists(TEMP_DIR),
            "temp_dir_writable": os.access(TEMP_DIR, os.W_OK) if os.path.exists(TEMP_DIR) else False
        }

        # Model status
        if model_loaded:
            health_status["model_status"] = {
                "loaded": True,
                "device": str(device) if device else "unknown"
            }
        else:
            health_status["model_status"] = {
                "loaded": False,
                "device": None
            }

        return jsonify(health_status), 200

    except Exception as e:
        error_response = {
            "status": "unhealthy",
            "service": "drishya",
            "error": str(e),
            "timestamp": time.time()
        }
        return jsonify(error_response), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload"""
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and process the image
        image = Image.open(file.stream)

        # Ensure image is in RGB format for consistent processing
        if image.mode == 'RGBA':
            # Convert RGBA to RGB by compositing over white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        image_np = np.array(image)

        # Resize image if too large (max 1920px width to prevent memory issues)
        max_width = 1920
        if image_np.shape[1] > max_width:
            scale_factor = max_width / image_np.shape[1]
            new_width = max_width
            new_height = int(image_np.shape[0] * scale_factor)
            image_resized = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            image_resized = image_np

        # Clean up old files
        cleanup_old_files()

        # Store original and resized image as temporary files
        # Store original for processing
        img_buffer_orig = BytesIO()
        Image.fromarray(image_np).save(img_buffer_orig, format='PNG')
        orig_filename = save_temp_image(img_buffer_orig.getvalue(), 'original')

        # Store resized for display (smaller file size)
        img_buffer_display = BytesIO()
        Image.fromarray(image_resized).save(img_buffer_display, format='JPEG', quality=85)
        display_filename = save_temp_image(img_buffer_display.getvalue(), 'display')

        session['original_image_file'] = orig_filename
        session['display_image_file'] = display_filename
        session['original_shape'] = image_np.shape
        session['display_shape'] = image_resized.shape

        return jsonify({
            'success': True,
            'width': image_resized.shape[1],
            'height': image_resized.shape[0],
            'original_width': image_np.shape[1],
            'original_height': image_np.shape[0]
        })

    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

@app.route('/get_image')
def get_image():
    """Serve the uploaded image"""
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    if 'display_image_file' not in session:
        return jsonify({'error': 'No image uploaded'}), 400

    filepath = get_temp_image_path(session['display_image_file'])
    if not os.path.exists(filepath):
        return jsonify({'error': 'Image file not found'}), 404

    return send_file(filepath, mimetype='image/jpeg')

@app.route('/generate_mask', methods=['POST'])
def generate_mask():
    """Generate mask from bounding box"""
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    if 'original_image_file' not in session:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        # Get bounding box coordinates
        data = request.get_json()
        x_min = int(data['x_min'])
        y_min = int(data['y_min'])
        x_max = int(data['x_max'])
        y_max = int(data['y_max'])

        # Validate coordinates
        if x_min >= x_max or y_min >= y_max:
            return jsonify({'error': 'Invalid bounding box coordinates'}), 400

        # Load original image from file
        orig_filepath = get_temp_image_path(session['original_image_file'])
        if not os.path.exists(orig_filepath):
            return jsonify({'error': 'Original image file not found'}), 404

        image = Image.open(orig_filepath)
        image_np = np.array(image)

        # Ensure model is loaded
        global mask_predictor, model_loaded
        if not model_loaded:
            try:
                load_model()
            except Exception as e:
                return jsonify({'error': f'Failed to load model: {str(e)}'}), 500

        # Set the image for the predictor
        mask_predictor.set_image(image_np)

        # Generate masks with error handling
        masks, scores, logits = mask_predictor.predict(
            box=np.array([x_min, y_min, x_max, y_max]),
            multimask_output=True
        )

        if len(masks) == 0:
            return jsonify({'error': 'Failed to generate mask'}), 500

        # Get best mask by score
        best_mask_idx = np.argmax(scores)
        binary_mask = masks[best_mask_idx].astype(np.uint8) * 255

        # Validate mask
        if np.sum(binary_mask) == 0:
            return jsonify({'error': 'Generated mask is empty'}), 500

        # Store mask as temporary file
        mask_buffer = BytesIO()
        Image.fromarray(binary_mask).save(mask_buffer, format='PNG')
        mask_filename = save_temp_image(mask_buffer.getvalue(), 'mask')
        session['generated_mask_file'] = mask_filename

        # Create mask visualization (use display image for smaller size)
        display_filepath = get_temp_image_path(session['display_image_file'])
        display_image = Image.open(display_filepath)
        display_np = np.array(display_image)

        # Resize mask to match display image
        display_shape = session['display_shape']
        original_shape = session['original_shape']

        if display_shape != original_shape[:2]:
            mask_resized = cv2.resize(binary_mask, (display_shape[1], display_shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = binary_mask

        mask_vis = show_mask(mask_resized, display_np)

        # Store visualization as temporary file
        vis_buffer = BytesIO()
        Image.fromarray(mask_vis).save(vis_buffer, format='JPEG', quality=85)
        vis_filename = save_temp_image(vis_buffer.getvalue(), 'mask_vis')
        session['mask_visualization_file'] = vis_filename

        return jsonify({
            'success': True
        })

    except Exception as e:
        return jsonify({'error': f'Failed to generate mask: {str(e)}'}), 500

@app.route('/get_mask_visualization')
def get_mask_visualization():
    """Serve the mask visualization"""
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    if 'mask_visualization_file' not in session:
        return jsonify({'error': 'No mask generated'}), 400

    filepath = get_temp_image_path(session['mask_visualization_file'])
    if not os.path.exists(filepath):
        return jsonify({'error': 'Mask visualization file not found'}), 404

    return send_file(filepath, mimetype='image/jpeg')

@app.route('/replace_product', methods=['POST'])
def replace_product():
    """Replace product in image"""
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    if 'original_image_file' not in session or 'generated_mask_file' not in session:
        return jsonify({'error': 'Missing image or mask data'}), 400

    if 'product_image' not in request.files:
        return jsonify({'error': 'No product image provided'}), 400

    try:
        # Get parameters
        scale_factor = float(request.form.get('scale_factor', 1.0))
        use_blending = request.form.get('use_blending', 'true').lower() == 'true'
        feather_amount = int(request.form.get('feather_amount', 15))
        enable_color_grading = request.form.get('enable_color_grading', 'true').lower() == 'true'
        color_grade_strength = float(request.form.get('color_grade_strength', 0.5))
        grading_method = request.form.get('grading_method', 'Match Target Area')

        # Debug logging
        print(f"🔧 Scale factor received: {scale_factor}")
        print(f"🔧 Use blending: {use_blending}")
        print(f"🔧 Feather amount: {feather_amount}")

        # Load original image
        orig_filepath = get_temp_image_path(session['original_image_file'])
        if not os.path.exists(orig_filepath):
            return jsonify({'error': 'Original image file not found'}), 404

        original_image = Image.open(orig_filepath)
        original_np = np.array(original_image)

        # Load mask
        mask_filepath = get_temp_image_path(session['generated_mask_file'])
        if not os.path.exists(mask_filepath):
            return jsonify({'error': 'Mask file not found'}), 404

        mask_image = Image.open(mask_filepath)
        mask_np = np.array(mask_image)

        # Load product image
        product_file = request.files['product_image']
        product_image = Image.open(product_file.stream)

        # Ensure consistent image format handling
        if product_image.mode == 'RGBA':
            # Keep RGBA for products that have transparency
            product_np = np.array(product_image)
        elif product_image.mode != 'RGB':
            # Convert other formats to RGB
            product_image = product_image.convert('RGB')
            product_np = np.array(product_image)
        else:
            product_np = np.array(product_image)

        # Ensure new product is in the correct format
        if len(product_np.shape) == 2:  # Grayscale
            product_np = cv2.cvtColor(product_np, cv2.COLOR_GRAY2RGB)

        # Apply color grading if enabled
        graded_product = product_np.copy()
        if enable_color_grading:
            try:
                if grading_method == "Match Target Area":
                    graded_product = apply_color_grading(
                        product_np,
                        original_np,
                        mask_np,
                        color_grade_strength
                    )
                else:
                    full_mask = np.ones(original_np.shape[:2], dtype=np.uint8) * 255
                    graded_product = apply_color_grading(
                        product_np,
                        original_np,
                        full_mask,
                        color_grade_strength
                    )
            except Exception as e:
                print(f"Color grading failed: {str(e)}. Using original product colors.")

        # Replace the product
        result_image = replace_product_in_image(
            original_np,
            graded_product,
            mask_np,
            scale_factor,
            feather_amount,
            use_blending
        )

        # Store result as temporary file
        result_buffer = BytesIO()
        Image.fromarray(np.clip(result_image, 0, 255).astype(np.uint8)).save(result_buffer, format='PNG')
        result_filename = save_temp_image(result_buffer.getvalue(), 'result')
        session['result_image_file'] = result_filename

        return jsonify({
            'success': True
        })

    except Exception as e:
        return jsonify({'error': f'Failed to replace product: {str(e)}'}), 500

@app.route('/get_result_image')
def get_result_image():
    """Serve the result image"""
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    if 'result_image_file' not in session:
        return jsonify({'error': 'No result image available'}), 400

    filepath = get_temp_image_path(session['result_image_file'])
    if not os.path.exists(filepath):
        return jsonify({'error': 'Result image file not found'}), 404

    return send_file(filepath, mimetype='image/png')

@app.route('/download_result')
def download_result():
    """Download the result image"""
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    if 'result_image_file' not in session:
        return jsonify({'error': 'No result image available'}), 400

    filepath = get_temp_image_path(session['result_image_file'])
    if not os.path.exists(filepath):
        return jsonify({'error': 'Result image file not found'}), 404

    return send_file(
        filepath,
        mimetype='image/png',
        as_attachment=True,
        download_name='product_replaced_image.png'
    )

# Load model on application startup (works for all deployment methods)
try:
    print("Downloading model weights")
    load_model()
    print("model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    print("The app will still start, but model loading will be attempted on first use.")

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
