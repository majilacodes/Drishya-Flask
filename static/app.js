// Global variables
let uploadedImageData = null;
let currentBoundingBox = null;
let isDrawing = false;
let startX, startY;
let canvas, ctx;
let imageElement;
let productUploaded = false;
let maskGenerated = false;
let isProcessing = false;
let autoReplaceTimeout = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    setupCanvas();
});

function initializeEventListeners() {
    // Image upload
    const imageUpload = document.getElementById('imageUpload');
    const imageInput = document.getElementById('imageInput');
    
    imageUpload.addEventListener('click', () => imageInput.click());
    imageUpload.addEventListener('dragover', handleDragOver);
    imageUpload.addEventListener('drop', handleDrop);
    imageInput.addEventListener('change', handleImageUpload);
    
    // Product upload
    const productUpload = document.getElementById('productUpload');
    const productInput = document.getElementById('productInput');
    
    productUpload.addEventListener('click', () => productInput.click());
    productInput.addEventListener('change', handleProductUpload);
    
    // Buttons
    document.getElementById('generateMaskBtn').addEventListener('click', generateMask);
    document.getElementById('clearBoxBtn').addEventListener('click', clearBoundingBox);
    document.getElementById('replaceProductBtn').addEventListener('click', replaceProduct);
    document.getElementById('downloadBtn').addEventListener('click', downloadResult);
    
    // Sliders - with auto-replacement
    document.getElementById('scaleSlider').addEventListener('input', function() {
        updateScaleValue();
        autoReplaceProduct();
    });
    document.getElementById('featherSlider').addEventListener('input', function() {
        updateFeatherValue();
        autoReplaceProduct();
    });
    document.getElementById('colorStrengthSlider').addEventListener('input', function() {
        updateColorStrengthValue();
        autoReplaceProduct();
    });

    // Checkboxes - with auto-replacement
    document.getElementById('useBlending').addEventListener('change', function() {
        toggleFeatherControls();
        autoReplaceProduct();
    });
    document.getElementById('enableColorGrading').addEventListener('change', function() {
        toggleColorGradingControls();
        autoReplaceProduct();
    });

    // Radio buttons - with auto-replacement
    document.querySelectorAll('input[name="gradingMethod"]').forEach(radio => {
        radio.addEventListener('change', autoReplaceProduct);
    });
}

function setupCanvas() {
    canvas = document.getElementById('drawingCanvas');
    ctx = canvas.getContext('2d');
    
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleImageFile(files[0]);
    }
}

function handleImageUpload(e) {
    const file = e.target.files[0];
    if (file) {
        handleImageFile(file);
    }
}

function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        showMessage('Please select a valid image file.', 'error');
        return;
    }
    
    showLoading('Uploading image...');
    
    const formData = new FormData();
    formData.append('image', file);
    
    fetch('/upload_image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.success) {
            uploadedImageData = data;
            displayUploadedImage(data);
            activateStep(2);
        } else {
            showMessage(data.error || 'Failed to upload image', 'error');
        }
    })
    .catch(error => {
        hideLoading();
        showMessage('Error uploading image: ' + error.message, 'error');
    });
}

function displayUploadedImage(data) {
    const imagePreview = document.getElementById('imagePreview');
    const uploadedImage = document.getElementById('uploadedImage');

    // Add cache-busting timestamp
    const timestamp = new Date().getTime();
    uploadedImage.src = '/get_image?' + timestamp;
    imagePreview.classList.remove('hidden');

    // Setup canvas for drawing
    setupDrawingCanvas(data);
}

function setupDrawingCanvas(imageData) {
    const canvasContainer = document.getElementById('canvasContainer');
    canvasContainer.classList.remove('hidden');

    // Calculate canvas size (max 800px width)
    const maxWidth = 800;
    const aspectRatio = imageData.height / imageData.width;
    const canvasWidth = Math.min(maxWidth, imageData.width);
    const canvasHeight = canvasWidth * aspectRatio;

    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    // Load and draw the image on canvas
    imageElement = new Image();
    imageElement.onload = function() {
        ctx.drawImage(imageElement, 0, 0, canvasWidth, canvasHeight);
    };
    // Add cache-busting timestamp
    const timestamp = new Date().getTime();
    imageElement.src = '/get_image?' + timestamp;
}

function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    
    // Clear previous box
    clearBoundingBox();
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;
    
    // Redraw image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
    
    // Draw bounding box
    ctx.strokeStyle = 'rgba(255, 0, 0, 1)';
    ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
    ctx.lineWidth = 2;
    
    const width = currentX - startX;
    const height = currentY - startY;
    
    ctx.fillRect(startX, startY, width, height);
    ctx.strokeRect(startX, startY, width, height);
    
    // Update coordinates display
    updateCoordinatesDisplay(startX, startY, currentX, currentY);
}

function stopDrawing(e) {
    if (!isDrawing) return;
    isDrawing = false;
    
    const rect = canvas.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    
    // Calculate actual image coordinates (use original dimensions)
    const scaleX = uploadedImageData.original_width / canvas.width;
    const scaleY = uploadedImageData.original_height / canvas.height;
    
    const x_min = Math.min(startX, endX) * scaleX;
    const y_min = Math.min(startY, endY) * scaleY;
    const x_max = Math.max(startX, endX) * scaleX;
    const y_max = Math.max(startY, endY) * scaleY;
    
    if (Math.abs(endX - startX) > 10 && Math.abs(endY - startY) > 10) {
        currentBoundingBox = {
            x_min: Math.round(x_min),
            y_min: Math.round(y_min),
            x_max: Math.round(x_max),
            y_max: Math.round(y_max)
        };
        
        document.getElementById('generateMaskBtn').disabled = false;
        updateCoordinatesDisplay(startX, startY, endX, endY, true);
    }
}

function updateCoordinatesDisplay(x1, y1, x2, y2, final = false) {
    const coordinates = document.getElementById('coordinates');
    if (final && currentBoundingBox) {
        coordinates.textContent = `Box Coordinates: X(${currentBoundingBox.x_min}, ${currentBoundingBox.x_max}), Y(${currentBoundingBox.y_min}, ${currentBoundingBox.y_max})`;
    } else {
        coordinates.textContent = `Drawing: (${Math.round(x1)}, ${Math.round(y1)}) to (${Math.round(x2)}, ${Math.round(y2)})`;
    }
}

function clearBoundingBox() {
    currentBoundingBox = null;
    document.getElementById('generateMaskBtn').disabled = true;
    document.getElementById('coordinates').textContent = 'Click and drag to draw a bounding box';
    
    if (imageElement && imageElement.complete) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
    }
}

function generateMask() {
    if (!currentBoundingBox) {
        showMessage('Please draw a bounding box first.', 'error');
        return;
    }
    
    showLoading('Generating mask...');
    
    fetch('/generate_mask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(currentBoundingBox)
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.success) {
            maskGenerated = true;
            displayMaskResults();
            activateStep(3);
            activateStep(4);
        } else {
            showMessage(data.error || 'Failed to generate mask', 'error');
        }
    })
    .catch(error => {
        hideLoading();
        showMessage('Error generating mask: ' + error.message, 'error');
    });
}

function displayMaskResults() {
    const maskResults = document.getElementById('maskResults');
    const originalForMask = document.getElementById('originalForMask');
    const maskVisualization = document.getElementById('maskVisualization');

    // Add cache-busting timestamp
    const timestamp = new Date().getTime();
    originalForMask.src = '/get_image?' + timestamp;
    maskVisualization.src = '/get_mask_visualization?' + timestamp;

    maskResults.classList.remove('hidden');
}

function handleProductUpload(e) {
    const file = e.target.files[0];
    if (file) {
        if (!file.type.startsWith('image/')) {
            showMessage('Please select a valid image file.', 'error');
            return;
        }
        
        const productPreview = document.getElementById('productPreview');
        const productImage = document.getElementById('productImage');
        
        const reader = new FileReader();
        reader.onload = function(e) {
            productImage.src = e.target.result;
            productPreview.classList.remove('hidden');

            // Show controls and replace button
            document.getElementById('replacementControls').style.display = 'grid';
            document.getElementById('replaceProductBtn').style.display = 'inline-block';

            // Mark product as uploaded and trigger initial replacement
            productUploaded = true;
            if (maskGenerated) {
                // Auto-replace immediately when product is first uploaded
                setTimeout(() => {
                    window.isAutoReplace = true;
                    replaceProduct();
                    window.isAutoReplace = false;
                }, 100);
            }
        };
        reader.readAsDataURL(file);
    }
}

function replaceProduct() {
    const productInput = document.getElementById('productInput');
    if (!productInput.files[0]) {
        showMessage('Please upload a product image first.', 'error');
        return;
    }

    if (isProcessing) {
        return; // Prevent multiple simultaneous requests
    }

    isProcessing = true;
    showLoading('Replacing product...');

    const formData = new FormData();
    formData.append('product_image', productInput.files[0]);
    formData.append('scale_factor', document.getElementById('scaleSlider').value);
    formData.append('use_blending', document.getElementById('useBlending').checked);
    formData.append('feather_amount', document.getElementById('featherSlider').value);
    formData.append('enable_color_grading', document.getElementById('enableColorGrading').checked);
    formData.append('color_grade_strength', document.getElementById('colorStrengthSlider').value);

    const gradingMethod = document.querySelector('input[name="gradingMethod"]:checked').value;
    formData.append('grading_method', gradingMethod);

    fetch('/replace_product', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        isProcessing = false;
        if (data.success) {
            displayFinalResults();
            document.getElementById('resultsStep').style.display = 'block';
            // Only scroll on manual button click, not auto-replace
            if (!window.isAutoReplace) {
                document.getElementById('resultsStep').scrollIntoView({ behavior: 'smooth' });
            }
        } else {
            showMessage(data.error || 'Failed to replace product', 'error');
        }
    })
    .catch(error => {
        hideLoading();
        isProcessing = false;
        showMessage('Error replacing product: ' + error.message, 'error');
    });
}

function autoReplaceProduct() {
    // Only auto-replace if product is uploaded and mask is generated
    if (!productUploaded || !maskGenerated || isProcessing) {
        return;
    }

    // Clear any existing timeout
    if (autoReplaceTimeout) {
        clearTimeout(autoReplaceTimeout);
    }

    // Debounce the auto-replace to avoid too many requests
    autoReplaceTimeout = setTimeout(() => {
        window.isAutoReplace = true;
        showLoading('Updating preview...');
        replaceProduct();
        window.isAutoReplace = false;
    }, 500); // Wait 500ms after user stops adjusting controls
}

function displayFinalResults() {
    const finalOriginal = document.getElementById('finalOriginal');
    const finalResult = document.getElementById('finalResult');

    // Add cache-busting timestamp to force browser to reload the image
    const timestamp = new Date().getTime();
    finalOriginal.src = '/get_image?' + timestamp;
    finalResult.src = '/get_result_image?' + timestamp;
}

function downloadResult() {
    const link = document.createElement('a');
    link.href = '/download_result';
    link.download = 'product_replaced_image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Utility functions
function activateStep(stepNumber) {
    document.getElementById(`step${stepNumber}`).classList.add('active');
}

function updateScaleValue() {
    const slider = document.getElementById('scaleSlider');
    document.getElementById('scaleValue').textContent = slider.value;
}

function updateFeatherValue() {
    const slider = document.getElementById('featherSlider');
    document.getElementById('featherValue').textContent = slider.value;
}

function updateColorStrengthValue() {
    const slider = document.getElementById('colorStrengthSlider');
    document.getElementById('colorStrengthValue').textContent = slider.value;
}

function toggleFeatherControls() {
    const useBlending = document.getElementById('useBlending').checked;
    const featherControls = document.getElementById('featherControls');
    featherControls.style.display = useBlending ? 'block' : 'none';
}

function toggleColorGradingControls() {
    const enableColorGrading = document.getElementById('enableColorGrading').checked;
    const colorGradingControls = document.getElementById('colorGradingControls');
    colorGradingControls.style.display = enableColorGrading ? 'block' : 'none';
}

function showLoading(message) {
    const loadingIndicator = document.getElementById('loadingIndicator');
    const loadingText = document.getElementById('loadingText');
    loadingText.textContent = message;
    loadingIndicator.classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingIndicator').classList.add('hidden');
}

function showMessage(message, type) {
    const messageContainer = document.getElementById('messageContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = type;
    messageDiv.textContent = message;
    
    messageContainer.appendChild(messageDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.parentNode.removeChild(messageDiv);
        }
    }, 5000);
}
