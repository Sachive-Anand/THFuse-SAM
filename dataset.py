import random
import os

def load_dataset(image_paths, BATCH_SIZE, num_imgs=None, validate_masks=True, mask_base_path="/content/drive/MyDrive/sam_masks/"):
    """
    Load dataset with optional mask validation
    
    Args:
        image_paths: List of visible image paths
        BATCH_SIZE: Batch size for training
        num_imgs: Number of images to use (None for all)
        validate_masks: Whether to validate that masks exist
        mask_base_path: Base path for mask files
    
    Returns:
        original_imgs_path: Filtered list of image paths
        batches: Number of batches
    """
    if num_imgs is None:
        num_imgs = len(image_paths)
    
    # Use only the specified number of images
    original_imgs_path = image_paths[:num_imgs]
    
    # Filter images to only include those with masks if validation is enabled
    if validate_masks:
        original_imgs_path = validate_masks_exist(original_imgs_path, mask_base_path)
        if len(original_imgs_path) == 0:
            raise ValueError("No images with valid masks found. Cannot create dataset.")
    
    # Shuffle the dataset
    random.shuffle(original_imgs_path)
    
    # Handle batch size remainder
    mod = len(original_imgs_path) % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % len(original_imgs_path))
    print('Train images samples %s.' % str(len(original_imgs_path) / BATCH_SIZE))
    
    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    
    # Print dataset statistics
    print_dataset_stats(original_imgs_path, mask_base_path)
    
    return original_imgs_path, batches

def validate_masks_exist(image_paths, mask_base_path):
    """
    Validate that each visible image has a corresponding mask file
    
    Args:
        image_paths: List of visible image paths
        mask_base_path: Base path for mask files
    
    Returns:
        List of image paths that have corresponding masks
    """
    valid_paths = []
    missing_masks = []
    
    for img_path in image_paths:
        # Convert visible image path to mask path
        mask_path = img_path.replace('/content/kaist-dataset/', mask_base_path)
        mask_path = mask_path.replace('.jpg', '.png')
        
        if os.path.exists(mask_path):
            valid_paths.append(img_path)
        else:
            missing_masks.append(mask_path)
    
    print(f"Found {len(valid_paths)} images with masks")
    print(f"Found {len(missing_masks)} images without masks")
    
    # Save list of missing masks for debugging
    if missing_masks:
        os.makedirs('/content/debug', exist_ok=True)
        with open('/content/debug/missing_masks.txt', 'w') as f:
            for mask_path in missing_masks:
                f.write(f"{mask_path}\n")
        print(f"Missing masks list saved to /content/debug/missing_masks.txt")
    
    return valid_paths

def print_dataset_stats(image_paths, mask_base_path):
    """
    Print statistics about the dataset
    
    Args:
        image_paths: List of image paths
        mask_base_path: Base path for mask files
    """
    if not image_paths:
        print("Dataset is empty!")
        return
    
    # Count images by set
    set_counts = {}
    for path in image_paths:
        set_name = path.split('/')[0]  # e.g., 'set00'
        set_counts[set_name] = set_counts.get(set_name, 0) + 1
    
    print("\nDataset Statistics:")
    print(f"Total images: {len(image_paths)}")
    print("Images per set:")
    for set_name, count in sorted(set_counts.items()):
        print(f"  {set_name}: {count} images")
    
    # Check mask file sizes (sample a few)
    sample_masks = []
    for i, path in enumerate(image_paths[:min(5, len(image_paths))]):
        mask_path = path.replace('/content/kaist-dataset/', mask_base_path).replace('.jpg', '.png')
        sample_masks.append(mask_path)
    
    print("\nSample mask info:")
    for mask_path in sample_masks:
        if os.path.exists(mask_path):
            file_size = os.path.getsize(mask_path) / 1024  # KB
            print(f"  {os.path.basename(mask_path)}: {file_size:.1f} KB")
        else:
            print(f"  {os.path.basename(mask_path)}: MISSING")

def create_balanced_dataset(image_paths, mask_base_path, max_per_set=None):
    """
    Create a balanced dataset with equal representation from each set
    
    Args:
        image_paths: List of all image paths
        mask_base_path: Base path for mask files
        max_per_set: Maximum number of images per set (None for no limit)
    
    Returns:
        Balanced list of image paths
    """
    # Group images by set
    set_groups = {}
    for path in image_paths:
        set_name = path.split('/')[0]
        if set_name not in set_groups:
            set_groups[set_name] = []
        set_groups[set_name].append(path)
    
    # Balance the dataset
    balanced_paths = []
    min_count = min(len(paths) for paths in set_groups.values()) if max_per_set is None else max_per_set
    
    for set_name, paths in set_groups.items():
        # Filter paths that have masks
        valid_paths = validate_masks_exist(paths, mask_base_path)
        
        # Take up to min_count or max_per_set
        count = min_count if max_per_set is None else min(max_per_set, len(valid_paths))
        if count > 0:
            balanced_paths.extend(valid_paths[:count])
    
    random.shuffle(balanced_paths)
    print(f"Created balanced dataset with {len(balanced_paths)} images")
    
    return balanced_paths

def get_image_mask_pairs(image_paths, mask_base_path):
    """
    Get pairs of image paths and their corresponding mask paths
    
    Args:
        image_paths: List of visible image paths
        mask_base_path: Base path for mask files
    
    Returns:
        List of (image_path, mask_path) tuples
    """
    pairs = []
    for img_path in image_paths:
        mask_path = img_path.replace('/content/kaist-dataset/', mask_base_path)
        mask_path = mask_path.replace('.jpg', '.png')
        pairs.append((img_path, mask_path))
    
    return pairs

def split_dataset(image_paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        image_paths: List of image paths
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    
    Returns:
        train_paths, val_paths, test_paths
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError("Ratios must sum to 1.0")
    
    total = len(image_paths)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_paths = image_paths[:train_end]
    val_paths = image_paths[train_end:val_end]
    test_paths = image_paths[val_end:]
    
    print(f"Dataset split: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
    
    return train_paths, val_paths, test_paths
