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
from loss_p import VggDeep,VggShallow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(image_lists):
    image_mode = 'L'
    fusemodel = Fusenet()
    vgg_ir_model = VggDeep()
    vgg_vi_model = VggShallow()

    mse_loss = torch.nn.MSELoss()
    TVLoss = TV_Loss()
    L1_loss = nn.L1Loss()
    semantic_loss_fn = nn.L1Loss()

    fusemodel.cuda()
    vgg_ir_model.cuda()
    vgg_vi_model.cuda()

    tbar = trange(args.epochs, ncols=150)
    print('Start training.....')

    all_ssim_loss = 0.
    all_model_loss = 0.
    all_ir_feature_loss = 0.
    all_vi_feature_loss = 0.
    all_semantic_loss = 0.
    save_num = 0

    for e in tbar:
        print('Epoch %d.....' % (e + 1))
        image_set, batches = dataset.load_dataset(image_lists, args.batch_size)
        fusemodel.train()
        count = 0
        for batch in range(batches):
            image_paths = image_set[batch * args.batch_size:(batch * args.batch_size + args.batch_size)]
            
            base_dir = "/content/kaist-dataset/"
            mask_dir = "/content/drive/MyDrive/sam_masks/"

            paths_vi = []
            paths_ir = []
            paths_mask = []

            for path in image_paths:
                paths_vi.append(join(base_dir, path))
                paths_ir.append(join(base_dir, path.replace('visible', 'lwir')))
                paths_mask.append(join(mask_dir, path.replace('.jpg', '.png')))

            img_vi = utils.get_train_images_auto(paths_vi, height=args.image_height, width=args.image_width, mode=image_mode)
            img_ir = utils.get_train_images_auto(paths_ir, height=args.image_height, width=args.image_width, mode=image_mode)
            img_mask = utils.get_train_masks_auto(paths_mask, height=args.image_height, width=args.image_width, mode='L')

            count += 1

            optimizer_model = Adam(fusemodel.parameters(), args.learning_rate)
            optimizer_model.zero_grad()
            optimizer_vgg_ir = Adam(vgg_ir_model.parameters(), args.learning_rate_d)
            optimizer_vgg_ir.zero_grad()
            optimizer_vgg_vi = Adam(vgg_vi_model.parameters(), args.learning_rate_d)
            optimizer_vgg_vi.zero_grad()

            img_vi = img_vi.cuda()
            img_ir = img_ir.cuda()
            img_mask = img_mask.cuda()

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
                    time.ctime(), e + 1, count, batches,
                    all_model_loss / args.log_interval,
                    all_semantic_loss / args.log_interval)
                tbar.set_description(mesg)

                all_ssim_loss = 0.
                all_semantic_loss = 0.
            
    fusemodel.eval()
    fusemodel.cpu()
    save_model_filename = "Final_epoch_" + str(args.epochs) + ".model"
    save_model_path = os.path.join(args.save_model_path, save_model_filename)
    torch.save(fusemodel.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)

def main():
    kaist_dataset_path = "/content/kaist-dataset/"
    print(f"Searching for visible images in: {kaist_dataset_path}")

    visible_image_paths = []
    for root, _, files in os.walk(kaist_dataset_path):
        if 'visible' in root:
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    visible_image_paths.append(os.path.relpath(full_path, kaist_dataset_path))
    
    # This line limits the training to 1000 images for a quick test.
    # For the final training, you should comment out or remove this line.
    train_num = 1000
    print(f"Found {len(visible_image_paths)} visible images. Limiting training to {train_num}.")
    
    train_num = min(train_num, len(visible_image_paths))
    
    final_image_list = visible_image_paths[:train_num]
    random.shuffle(final_image_list)
    
    train(final_image_list)

if __name__ == "__main__":
    main()