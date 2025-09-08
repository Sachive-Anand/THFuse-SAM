import args
import time
import random
import torch
import torch.nn as nn
import utils
import dataset
from fusenet import Fusenet
from tqdm import tqdm, trange
from torch.optim import Adam
from os.path import join
from loss import final_ssim, TV_Loss
from loss_p import VggDeep, VggShallow
import os

# Colab automatically handles GPU, so we don't need CUDA_VISIBLE_DEVICES
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def find_complete_sets_images():
    """Find images only from sets with complete mask coverage (set03 and set05)"""
    kaist_base = "/content/kaist-dataset/"
    mask_base = "/content/drive/MyDrive/sam_masks/"
    
    images_with_masks = []
    target_sets = ['set03', 'set05']  # Only use sets with 100% mask coverage
    
    for set_name in target_sets:
        set_path = os.path.join(kaist_base, set_name)
        if not os.path.exists(set_path):
            continue
            
        # Walk through all videos in this set
        for video in os.listdir(set_path):
            video_path = os.path.join(set_path, video, 'visible')
            if not os.path.exists(video_path):
                continue
                
            # Get all images in this visible folder
            for file in os.listdir(video_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(set_name, video, 'visible', file)
                    
                    # Check if mask exists (it should, since we're using complete sets)
                    mask_path = os.path.join(mask_base, img_path)
                    mask_path = mask_path.replace('.jpg', '.png').replace('.jpeg', '.png')
                    
                    if os.path.exists(mask_path):
                        images_with_masks.append(img_path)
                    else:
                        print(f"Warning: Missing mask for {img_path}")
    
    return images_with_masks

def train(image_lists):
    if not image_lists:
        print("ERROR: No images found for training.")
        return
        
    image_mode = 'L'
    fusemodel = Fusenet()
    vgg_ir_model = VggDeep()
    vgg_vi_model = VggShallow()

    mse_loss = torch.nn.MSELoss()
    TVLoss = TV_Loss()
    L1_loss = nn.L1Loss()
    semantic_loss_fn = nn.L1Loss()

    # Move models to GPU (Colab provides GPU automatically)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    fusemodel.to(device)
    vgg_ir_model.to(device)
    vgg_vi_model.to(device)

    tbar = trange(args.epochs, ncols=150)
    print('Start training.....')

    all_ssim_loss = 0.
    all_model_loss = 0.
    all_ir_feature_loss = 0.
    all_vi_feature_loss = 0.
    all_semantic_loss = 0.
    save_num = 0

    for epoch in tbar:
        print('Epoch %d.....' % (epoch + 1))
        image_set, batches = dataset.load_dataset(image_lists, args.batch_size, validate_masks=False)
        fusemodel.train()
        count = 0
        
        for batch in range(batches):
            image_paths = image_set[batch * args.batch_size:(batch * args.batch_size + args.batch_size)]
            
            # Define base directories
            base_dir = "/content/kaist-dataset/"
            mask_base_dir = "/content/drive/MyDrive/sam_masks/"
            
            paths_vi = []
            paths_ir = []
            paths_mask = []
            
            for path in image_paths:
                # Construct paths for visible images
                vi_path = join(base_dir, path)
                paths_vi.append(vi_path)
                
                # Construct paths for infrared images
                ir_path = path.replace('visible', 'lwir')
                ir_path = join(base_dir, ir_path)
                paths_ir.append(ir_path)
                
                # Construct mask path
                mask_path = path.replace('.jpg', '.png')
                mask_path = join(mask_base_dir, mask_path)
                paths_mask.append(mask_path)
            
            try:
                img_vi = utils.get_train_images_auto(paths_vi, height=args.image_height, width=args.image_width, mode=image_mode)
                img_ir = utils.get_train_images_auto(paths_ir, height=args.image_height, width=args.image_width, mode=image_mode)
                img_mask = utils.get_train_masks_auto(paths_mask, height=args.image_height, width=args.image_width, mode='L')
                
                # Check if any images failed to load
                if img_vi is None or img_ir is None or img_mask is None:
                    print(f"Skipping batch due to missing images")
                    continue
                    
            except Exception as e:
                print(f"Error loading images: {e}")
                continue

            count += 1

            optimizer_model = Adam(fusemodel.parameters(), args.learning_rate)
            optimizer_model.zero_grad()
            optimizer_vgg_ir = Adam(vgg_ir_model.parameters(), args.learning_rate_d)
            optimizer_vgg_ir.zero_grad()
            optimizer_vgg_vi = Adam(vgg_vi_model.parameters(), args.learning_rate_d)
            optimizer_vgg_vi.zero_grad()

            # Move data to the same device as models
            img_vi = img_vi.to(device)
            img_ir = img_ir.to(device)
            img_mask = img_mask.to(device)

            outputs = fusemodel(img_vi, img_ir, img_mask)

            ssim_loss_value = 1 - final_ssim(img_ir, img_vi, outputs)
            mse_loss_value = 0
            TV_loss_value = 0
            
            semantic_loss_value = semantic_loss_fn(outputs * img_mask, img_mask)
            
            model_loss = ssim_loss_value + (0.05 * mse_loss_value) + (0.05 * TV_loss_value) + (0.7 * semantic_loss_value)
            model_loss.backward()
            optimizer_model.step()

            vgg_ir_fuse_out = vgg_ir_model(outputs.detach())[2]
            vgg_ir_out = vgg_ir_model(img_ir)[2]
            per_loss_ir = L1_loss(vgg_ir_fuse_out, vgg_ir_out)
            per_loss_ir.backward()
            optimizer_vgg_ir.step()

            vgg_vi_fuse_out = vgg_vi_model(outputs.detach())[0]
            vgg_vi_out = vgg_vi_model(img_vi)[0]
            per_loss_vi = L1_loss(vgg_vi_fuse_out, vgg_vi_out)
            per_loss_vi.backward()
            optimizer_vgg_vi.step()

            all_ssim_loss += ssim_loss_value.item()
            all_model_loss = all_ssim_loss
            all_ir_feature_loss += per_loss_ir.item()
            all_vi_feature_loss += per_loss_vi.item()
            all_semantic_loss += semantic_loss_value.item()

            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:[{}/{}] fusemodel loss: {:.5f} semantic loss: {:.5f}".format(
                    time.ctime(), epoch + 1, count, batches,
                    all_model_loss / args.log_interval,
                    all_semantic_loss / args.log_interval)
                tbar.set_description(mesg)

                all_ssim_loss = 0.
                all_semantic_loss = 0.
            
    fusemodel.eval()
    # Move model back to CPU before saving
    fusemodel.cpu()
    save_model_filename = f"THFuse_SAM_epoch_{args.epochs}_{len(image_lists)}_images.model"
    save_model_path = os.path.join(args.save_model_path, save_model_filename)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_model_path, exist_ok=True)
    
    torch.save(fusemodel.state_dict(), save_model_path)
    print(f"\n✅ Training completed! Model saved at: {save_model_path}")
    
    # Also save to Google Drive for permanence
    drive_save_path = os.path.join("/content/drive/MyDrive/models/", save_model_filename)
    os.makedirs("/content/drive/MyDrive/models/", exist_ok=True)
    torch.save(fusemodel.state_dict(), drive_save_path)
    print(f"✅ Model also saved to Google Drive: {drive_save_path}")

def main():
    print("=" * 60)
    print("THFuse-SAM Training on Colab GPU")
    print("=" * 60)
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"🎯 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"🎯 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  No GPU detected. Training will be very slow on CPU.")
        print("⚠️  Please ensure you're using Colab with GPU runtime")
    
    print("Finding images from complete sets (set03 and set05)...")
    images_with_masks = find_complete_sets_images()
    
    if not images_with_masks:
        print("❌ No images found in complete sets!")
        return
    
    print(f"✅ Found {len(images_with_masks)} images from complete sets (set03 + set05)")
    
    # Use all available images or limit to train_num
    train_num = min(args.train_num, len(images_with_masks))
    final_image_list = images_with_masks[:train_num]
    random.shuffle(final_image_list)
    
    print(f"📊 Using {len(final_image_list)} images for training")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📊 Epochs: {args.epochs}")
    print(f"📊 Learning rate: {args.learning_rate}")
    print("=" * 60)
    
    # Train with these images
    train(final_image_list)

if __name__ == "__main__":
    main()
