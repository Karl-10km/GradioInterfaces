from io import BytesIO, BufferedReader
from typing import Optional, List, Dict, Tuple, Any
import base64

from PIL import Image

def crop_image_ratio(img, target_aspect_ratio: float) -> Image.Image:
    # Validate img parameter
    if not isinstance(img, Image.Image):
        raise TypeError("crop_image_ratio: img must be a PIL Image object")
    
    # Validate target_aspect_ratio parameter
    if not isinstance(target_aspect_ratio, (int, float)):
        raise TypeError("crop_image_ratio: target_aspect_ratio must be a number")
    
    if target_aspect_ratio <= 0:
        raise ValueError("crop_image_ratio: target_aspect_ratio must be a positive number")
    
    # Get original dimensions
    original_width, original_height = img.size

    # Calculate new dimensions for cropping
    if original_height / original_width < target_aspect_ratio:
        # Original image is wider than target, crop width
        new_width = int(original_height / target_aspect_ratio)
        new_height = original_height
    else:
        # Original image is taller than target, crop height
        new_height = int(original_width * target_aspect_ratio)
        new_width = original_width

    # Calculate crop box coordinates for center crop
    left = (original_width - new_width) / 2
    upper = (original_height - new_height) / 2
    right = left + new_width
    lower = upper + new_height

    # Ensure coordinates are integers
    left, upper, right, lower = int(left), int(upper), int(right), int(lower)

    # Perform the crop
    cropped_image = img.crop((left, upper, right, lower))
    return cropped_image

def crop_resize_image(img, target_aspect_ratio: float, target_size: tuple[int, int]) -> Image.Image:
    # Validate img parameter
    if not isinstance(img, Image.Image):
        raise TypeError("crop_resize_image: img must be a PIL Image object")
    
    # Validate target_size parameter
    if not isinstance(target_size, tuple):
        raise TypeError("crop_resize_image: target_size must be a tuple")
    
    if len(target_size) != 2:
        raise ValueError("crop_resize_image: target_size must be a tuple with exactly 2 elements (width, height)")
    
    width, height = target_size
    if not isinstance(width, int) or not isinstance(height, int):
        raise TypeError("crop_resize_image: target_size elements must be integers")
    
    if width <= 0 or height <= 0:
        raise ValueError("crop_resize_image: target_size elements must be positive integers")
    
    cropped_image = crop_image_ratio(img, target_aspect_ratio)
    resized_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)

    return resized_image

def byte2pil(buffer_reader) -> Image.Image:
    # Validate buffer_reader parameter
    if not hasattr(buffer_reader, 'read'):
        raise TypeError("byte2pil: buffer_reader must be a ReadableBuffer (object with read method)")
    
    try:
        byte_stream = BytesIO(buffer_reader.read())
        pil_output = Image.open(byte_stream)
        return pil_output
    except Exception as e:
        raise ValueError(f"byte2pil: Failed to convert buffer to PIL Image - {str(e)}")

def pil2byte(img: Image.Image, format: str = 'PNG'):
    # Validate img parameter
    if not isinstance(img, Image.Image):
        raise TypeError("pil2byte: img must be a PIL Image object")
    
    try:
        byte_stream = BytesIO()
        img.save(byte_stream, format=format.upper())
        byte_stream.seek(0)
        img_output = BufferedReader(byte_stream)
        return img_output
    except Exception as e:
        raise ValueError(f"pil2byte: Failed to convert PIL Image to ReadableBuffer - {str(e)}")


def base64encoder(img: Image.Image, format: str = 'PNG') -> str:
    """
    Convert PIL Image to base64 encoded string.
    
    Args:
        img: PIL Image object to encode
        format: Image format for encoding (default: 'PNG')
    
    Returns:
        Base64 encoded string of the image
    """
    # Validate img parameter
    if not isinstance(img, Image.Image):
        raise TypeError("base64encoder: img must be a PIL Image object")
    
    # Validate format parameter
    if not isinstance(format, str):
        raise TypeError("base64encoder: format must be a string")
    
    if not format.strip():
        raise ValueError("base64encoder: format cannot be empty")
    
    try:
        byte_stream = BytesIO()
        img.save(byte_stream, format=format.upper())
        byte_stream.seek(0)
        image_bytes = byte_stream.getvalue()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        return base64_string
    except Exception as e:
        raise ValueError(f"base64encoder: Failed to encode PIL Image to base64 - {str(e)}")
