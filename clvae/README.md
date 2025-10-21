# Contrastive Learning Variational Autoencoder
 This module creates a dataset by extracting frames from videos, selecting "good features to track" within each frame, then tracking these points with a traditional OpenCV tracker. After applying temporally cycle consistency check, we create patches from point quartets and end up with a sequence of each patch throughout each video, using the process described in [here](../tcc/README.md). The CL-VAE then leverages contrastive learning to learn robust feature representations of patches.

## Preparation

Generate the patch dataset from a folder containing videos of the setup you wish to train the models on. Use the instructions in [here](../tcc/README.md). 

## Training and evaluating model

```bash
python train_cl_vae.py --video_dir ../data/pickles/ --img_channels 3 --resize 16 16
```
### Arguments

#### Dataset Arguments

* `-vd/--video_dir` (required): Path to the directory containing video files.
* `-ps/--pair_step` (default: 5): Frame difference of positive pairs.
* `-rs/--resize` (default: [20, 20]): Model input size.

#### Model Arguments

* `-ld/--latent_dim` (default: 128): Dimension of the latent space.
* `-pd/--projection_dim` (default: 64): Dimension of the projection space for contrastive loss.
* `-ic/--img_channels` (default: 1): Number of image channels (default: 1 for grayscale images).

#### Training Arguments

* `-bs/--batch_size` (default: 32): Batch size for training.
* `-lr/--learning_rate` (default: 1e-3): Learning rate for the optimizer.
* `-e/--num_epochs` (default: 50): Number of epochs to train.
* `-a/--alpha` (default: 0.2): Weight for reconstruction loss.
* `-b/--beta` (default: 0.8): Weight for contrastive loss.
* `-rlt/--rec_loss_type` (default: 'mse'): Type of reconstruction loss (mse/ssim/bce).

#### Output Arguments

* `-od/--output_dir` (default: "../logging"): Directory to save model checkpoints.
* `-v/--visualize` (optional): Visualize the training pair batches.
* `-si/--save_interval` (default: 10): Interval (in epochs) to save model checkpoints.

## Results
After every training process a folder named by the current date and time is stored in the logging folder. This folder contains the following:

* **Loss Tracking**: Training and validation losses for each epoch, along with test loss, are saved to a CSV file
* **Model Checkpoints**: Checkpoints of the trained VAE model are saved 
* **Training Configuration**: Information about the training parameters provided are saved to a json file inside the folder
* **Testing Results**: A folder named 'tests' is created where the VAE reconstruction patches from the test set are saved as input-output images side-by-side