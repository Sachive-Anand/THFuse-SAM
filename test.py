import torch
from torch.autograd import Variable
import utils
import numpy as np
import time
from fusenet import Fusenet
import os
import cv2
from segment_anything import sam_model_registry, SamPredictor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model(path):
    fuse_net = Fusenet()
    fuse_net.load_state_dict(torch.load(path))
    para = sum([np.prod(list(p.size())) for p in fuse_net.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(fuse_net._get_name(), para * type_size / 1000 / 1000))
    fuse_net.eval()
    fuse_net.cuda()
    return fuse_net

def generate_fuse_image(model, vi, ir, mask):
    out = model(vi, ir, mask)
    return out

def fuse_test(model, predictor, vi_path, ir_path, output_path_root, index):
    vi_img_tensor = utils.get_test_images(vi_path, height=None, width=None)
    ir_img_tensor = utils.get_test_images(ir_path, height=None, width=None)
    out_shape_source = utils.get_image(vi_path, height=None, width=None)

    image_for_sam = cv2.imread(vi_path)
    image_for_sam = cv2.cvtColor(image_for_sam, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_for_sam)
    h, w, _ = image_for_sam.shape
    
    points_per_side = 16
    x = np.linspace(0, w - 1, points_per_side)
    y = np.linspace(0, h - 1, points_per_side)
    xv, yv = np.meshgrid(x, y)
    grid_points = np.stack([xv.flatten(), yv.flatten()], axis=1)
    
    masks, _, _ = predictor.predict(
        point_coords=grid_points,
        point_labels=np.ones(len(grid_points)),
        multimask_output=False,
    )
    
    combined_mask = np.zeros((h, w), dtype=np.float32)
    for mask in masks:
        combined_mask[mask] = 1.0

    mask_tensor = torch.from_numpy(combined_mask).unsqueeze(0).unsqueeze(0)

    vi_img_tensor = vi_img_tensor.cuda()
    ir_img_tensor = ir_img_tensor.cuda()
    mask_tensor = mask_tensor.cuda()

    vi_img_tensor = Variable(vi_img_tensor, requires_grad=False)
    ir_img_tensor = Variable(ir_img_tensor, requires_grad=False)
    mask_tensor = Variable(mask_tensor, requires_grad=False)

    img_fusion = generate_fuse_image(model, vi_img_tensor, ir_img_tensor, mask_tensor)

    file_name = f'fusion_{index}_{os.path.basename(vi_path)}'
    output_path = os.path.join(output_path_root, file_name)

    if torch.cuda.is_available():
        img = (img_fusion.cpu().data.numpy() * 255).clip(0, 255)
    else:
        img = (img_fusion.data.numpy() * 255).clip(0, 255)
        
    img = img.astype('uint8')
    utils.save_images(output_path, img, out_shape_source)
    print(f"Saved fused image to: {output_path}")

def main():
    # Define test cases based on the folders you have (e.g., set05)
    test_cases = [
        {'set': 'set05', 'video': 'V000', 'image': 'I00000.jpg'},
        {'set': 'set05', 'video': 'V001', 'image': 'I00100.jpg'},
        {'set': 'set04', 'video': 'V000', 'image': 'I00200.jpg'},
    ]
    
    base_dir = "/content/kaist-dataset/"
    output_path = "/content/results/"
    os.makedirs(output_path, exist_ok=True)
    
    model_path = "/content/Final_epoch_2.model" # Your trained model
    
    print("Initializing SAM model for inference...")
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("SAM model initialized.")
    
    with torch.no_grad():
        if not os.path.exists(model_path):
            print(f"Model file does not exist: {model_path}")
            return
            
        model = load_model(model_path)
        
        for i, case in enumerate(test_cases):
            # Construct the full paths for visible and infrared images
            visible_path = os.path.join(base_dir, case['set'], case['video'], 'visible', case['image'])
            infrared_path = os.path.join(base_dir, case['set'], case['video'], 'lwir', case['image'])
            
            print(f"\nProcessing image {i+1}/{len(test_cases)}: {visible_path}")
            
            if not os.path.exists(visible_path) or not os.path.exists(infrared_path):
                print("--- SKIPPING: One or both image paths do not exist. ---")
                continue

            start = time.time()
            fuse_test(model, predictor, visible_path, infrared_path, output_path, i + 1)
            end = time.time()
            print(f'Time: {end - start:.4f} S')
            
    print('\nDone......')

if __name__ == "__main__":
    main()