import argparse
import os
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import json
from datetime import datetime

import sys
import pathlib
add_path = pathlib.Path.absolute(pathlib.Path(__file__)).parent.parent.parent
sys.path.insert(0, str(add_path))
from clvae.modules.vae import CLVAE  
from clvae.utils.cl_dataset_utils import create_loaders


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader = create_loaders(args.video_dir, args.batch_size, args.pair_step, args.resize, args.visualize, args.output_dir)

    model = CLVAE(
        img_channels=args.img_channels,
        latent_dim=args.latent_dim,
        img_height=args.resize[0],
        img_width=args.resize[1],
        projection_dim=args.projection_dim
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    min_val_loss = float("inf")
    logging_df = pd.DataFrame(columns=["Epoch", "Total Training Loss", "Train Rec Loss", "Train CL Loss" , \
                                           "Total Validation Loss", "Val Rec Loss", "Val CL Loss"])
    
    writer = SummaryWriter(log_dir=args.output_dir)
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        train_rec_loss, train_cl_loss = (0.0, 0.0)
        
        for batch in train_loader:
            loss, rec_loss, cl_loss = model.train_step(model, batch, optimizer, device=device, alpha=args.alpha, beta=args.beta, rec_loss_type=args.rec_loss_type)
            writer.add_scalar('Train/Reconstruction Loss', rec_loss, global_step)
            writer.add_scalar('Train/Contrastive Loss', cl_loss, global_step)
            total_loss += loss
            train_rec_loss += rec_loss / len(train_loader)
            train_cl_loss += cl_loss / len(train_loader)
            global_step += 1

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar('Train/Total Loss', avg_train_loss, epoch)

        model.eval()
        total_val_loss = 0.0
        val_rec_loss, val_cl_loss = (0.0, 0.0)
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if args.log_rec and i == 0: 
                    # If you want to visualize reconstructions
                    log_params = dict(writer=writer, epoch=epoch, batch_idx=i)
                else:
                    log_params = None
                loss, rec_loss, cl_loss = model.val_step(model, batch, device=device, alpha=args.alpha, beta=args.beta, rec_loss_type=args.rec_loss_type, log_params=log_params)
                total_val_loss += loss
                val_rec_loss += rec_loss / len(val_loader)
                val_cl_loss += cl_loss / len(val_loader)

        avg_val_loss = total_val_loss / len(val_loader)

        model_info = {
            "epoch": epoch + 1,
            "val_loss": avg_val_loss,
            "model_state_dict": model.state_dict()
        }
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            save_path = os.path.join(args.output_dir, f"vae_best.pth")
            torch.save(model_info, save_path)

        torch.save(model_info, os.path.join(args.output_dir, f"vae_latest.pth"))
        
        writer.add_scalar('Val/Reconstruction Loss', val_rec_loss, global_step)
        writer.add_scalar('Val/Contrastive Loss', val_cl_loss, global_step)
        writer.add_scalar('Val/Total Loss', avg_val_loss, global_step)

        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(f"Train Rec Loss: {train_rec_loss:.4f}, Train CL Loss: {train_cl_loss:.4f} || Val Rec Loss: {val_rec_loss:.4f}, Val CL Loss: {val_cl_loss:.4f}")
        logging_df.loc[len(logging_df)+1] = [int(epoch + 1), avg_train_loss, train_rec_loss, train_cl_loss, avg_val_loss, val_rec_loss, val_cl_loss]
        logging_df.to_csv(os.path.join(args.output_dir, "training_logs.csv"), index=False)
        
    final_save_path = os.path.join(args.output_dir, f"vae_final.pth")
    model_info = {
        "epoch": epoch + 1,
        "val_loss": avg_val_loss,
        "model_state_dict": model.state_dict()
    }
    torch.save(model_info, final_save_path)
    print(f"Final model saved to {final_save_path}")

    logging_df.to_csv(os.path.join(args.output_dir, "training_logs.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CLVAE model for template matching with contrastive loss")

    # Dataset arguments
    parser.add_argument("-vd", "--video_dir", type=str, required=True, help="Path to the directory containing video files")
    parser.add_argument("-vvd", "--validation_video_dir", type=str, default=None, help="Path to the directory containing validation video files")
    parser.add_argument("-ps", "--pair_step", type=int, default=5, help="Frame difference of positive pairs")
    parser.add_argument("-rs", "--resize", type=int, nargs=2, default=[16, 16], help="Model input size")
    
    # Model arguments
    parser.add_argument("-ld", "--latent_dim", type=int, default=128, help="Dimension of the latent space")
    parser.add_argument("-pd", "--projection_dim", type=int, default=64, help="Dimension of the projection space for contrastive loss")
    parser.add_argument("-ic", "--img_channels", type=int, default=1, help="Number of image channels (default: 1 for grayscale images)")
    
    # Training arguments
    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer")
    parser.add_argument("-e", "--num_epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("-a", "--alpha", type=float, default=0.2, help="Weight for reconstruction loss")
    parser.add_argument("-b", "--beta", type=float, default=0.8, help="Weight for contrastive loss")
    parser.add_argument("-rlt", "--rec_loss_type", type=str, default="mse", help="Type of reconstruction loss (mse/ssim/bce)")

    # Output arguments
    parser.add_argument("-od", "--output_dir", type=str, default="../logging", help="Directory to save model checkpoints")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize the training pair batches")
    parser.add_argument("-si", "--save_interval", type=int, default=10, help="Interval (in epochs) to save model checkpoints")
    parser.add_argument("--log_rec", action="store_true", help="Log reconstructions for validation set")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"exp_{timestamp}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save training parameters to a JSON file
    params = vars(args)
    params_path = os.path.join(args.output_dir, "training_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

    main(args)
