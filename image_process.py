from PIL import Image

def crop_image_ratio(img, target_aspect_ratio):
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
