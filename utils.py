import torch
import numpy as np
from os import listdir
import cv2
from scipy import ndimage
from torchvision import transforms
from imageio import imread
import os

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(file)
        elif name.endswith('.jpg'):
            images.append(file)
        elif name.endswith('.jpeg'):
            images.append(file)
        name1 = name.split('.')
        names.append(name1[0])
    return images

def get_image(path, height=256, width=256, mode='L'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    try:
        if mode == 'L':
            image = imread(path, pilmode="L")
        else:
            image = imread(path, pilmode="RGB")
            
        if height is not None and width is not None:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return image
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if image is None:
            continue
            
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    if len(images) == 0:
        return None
        
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def get_test_images(paths, height=None, width=None, mode='L'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if image is None:
            continue
            
        w, h = image.shape[0], image.shape[1]
        w_s = 256 - w % 256
        h_s = 256 - h % 256
        image = cv2.copyMakeBorder(image, 0, w_s, 0, h_s, cv2.BORDER_CONSTANT,
                                     value=128)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    
    if len(images) == 0:
        return None
        
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def save_images(path, data, out):
    w, h = out.shape[0], out.shape[1]
    if data.shape[1] == 1:
        data = data.reshape([data.shape[2], data.shape[3]])
    ori = data[0:w, 0:h]
    cv2.imwrite(path, ori)

def get_train_masks_auto(paths, height=256, width=256, mode='L'):
    """
    This function loads a batch of mask images from a list of paths.
    It normalizes the mask values from [0, 255] to [0, 1] range.
    Also handles missing mask files gracefully.
    """
    if isinstance(paths, str):
        paths = [paths]
    images = []
    
    for path in paths:
        # Check if mask file exists
        if not os.path.exists(path):
            print(f"Warning: Mask file not found: {path}")
            # Create a blank mask as fallback
            blank_mask = np.zeros((height, width), dtype=np.float32)
            if mode == 'L':
                blank_mask = np.reshape(blank_mask, [1, height, width])
            else:
                blank_mask = np.reshape(blank_mask, [3, height, width])
            images.append(blank_mask)
            continue
            
        try:
            # Load the mask image
            mask = get_image(path, height, width, mode=mode)
            if mask is None:
                # Create blank mask if loading fails
                blank_mask = np.zeros((height, width), dtype=np.float32)
                if mode == 'L':
                    blank_mask = np.reshape(blank_mask, [1, height, width])
                else:
                    blank_mask = np.reshape(blank_mask, [3, height, width])
                images.append(blank_mask)
                continue
                
            # Normalize mask values from [0, 255] to [0, 1]
            mask = mask.astype(np.float32) / 255.0
            
            # Apply threshold to ensure binary-like masks (optional)
            # mask = (mask > 0.1).astype(np.float32)
            
            if mode == 'L':
                # Masks are single-channel
                mask = np.reshape(mask, [1, mask.shape[0], mask.shape[1]])
            else:
                # For multi-channel images (unlikely for masks)
                mask = np.reshape(mask, [mask.shape[2], mask.shape[0], mask.shape[1]])
                
            images.append(mask)
            
        except Exception as e:
            print(f"Error loading mask {path}: {e}")
            # Create blank mask as fallback
            blank_mask = np.zeros((height, width), dtype=np.float32)
            if mode == 'L':
                blank_mask = np.reshape(blank_mask, [1, height, width])
            else:
                blank_mask = np.reshape(blank_mask, [3, height, width])
            images.append(blank_mask)

    if len(images) == 0:
        print("Error: No masks could be loaded")
        return None
        
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def validate_image_mask_pairs(visible_paths, mask_base_path):
    """
    Validate that visible images have corresponding mask files
    Returns list of valid visible image paths
    """
    valid_paths = []
    
    for vi_path in visible_paths:
        # Construct corresponding mask path
        mask_path = vi_path.replace('/content/kaist-dataset/', mask_base_path)
        mask_path = mask_path.replace('.jpg', '.png')
        
        if os.path.exists(mask_path):
            valid_paths.append(vi_path)
        else:
            print(f"Missing mask for: {vi_path}")
            
    return valid_paths

def create_dummy_mask_if_missing(mask_path, height=256, width=256):
    """
    Create a dummy mask if the real mask is missing
    This ensures training can continue even with missing masks
    """
    if not os.path.exists(mask_path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        
        # Create a simple centered rectangle as dummy mask
        dummy_mask = np.zeros((height, width), dtype=np.uint8)
        center_y, center_x = height // 2, width // 2
        size = min(height, width) // 4
        cv2.rectangle(dummy_mask, 
                     (center_x - size, center_y - size),
                     (center_x + size, center_y + size),
                     255, -1)  # Filled rectangle
        
        # Save dummy mask
        cv2.imwrite(mask_path, dummy_mask)
        print(f"Created dummy mask: {mask_path}")
        
    return mask_path
