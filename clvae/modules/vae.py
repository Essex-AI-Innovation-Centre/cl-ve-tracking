import torch
import torch.nn as nn
import os
from datetime import datetime
import json
from pytorch_msssim import ssim
import torch.nn.functional as F
import torchvision

class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=128, img_height=16, img_width=16):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 16, 3, 2, 1),  # (batch, 16, img_height/2, img_width/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),           # (batch, 32, img_height/4, img_width/4)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (img_height//4) * (img_width//4), 256),  # Match flattened size
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)        # Mean and log variance
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * (img_height//4) * (img_width//4)),  # Map back to spatial dimensions
            nn.ReLU(),
            nn.Unflatten(1, (32, img_height//4, img_width//4)),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),  # (batch, 16, img_height/2, img_width/2)
            nn.ReLU(),
            nn.ConvTranspose2d(16, img_channels, 3, 2, 1, 1),  # (batch, img_channels, img_height, img_width)
            nn.Sigmoid()
        )

    def encode(self, x):
        params = self.encoder(x)
        mean, log_var = params.chunk(2, dim=-1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mean, log_var
    
    def calculate_rec_loss(self, recon_x, x, mean, log_var, alpha=0.8, loss_type='ssim'):
        if loss_type == 'bce':           
            recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
        elif loss_type == 'mse':
            recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        elif loss_type == 'ssim':
            recon_loss = 1 - ssim(recon_x, x, data_range=1, size_average=True)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return alpha * recon_loss + kl_divergence
    
    def train_step(self, model, batch, optimizer, device="cuda", alpha=0.8, loss_type='ssim'):
        model.train()
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mean, log_var = model(batch)
        loss = self.calculate_rec_loss(recon_batch, batch, mean, log_var, alpha=alpha, loss_type=loss_type)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def val_step(self, model, batch, device="cuda", alpha=0.8, loss_type='ssim'):
        model.eval()
        with torch.no_grad():
            batch = batch.to(device)
            recon_batch, mean, log_var = model(batch)
            loss = self.calculate_rec_loss(recon_batch, batch, mean, log_var, alpha=alpha, loss_type=loss_type)
        return loss.item()

    def run_inference(self, model, image, device="cuda"):
        model.eval()
        with torch.no_grad():
            image = image.to(device).unsqueeze(0)  # Add batch dimension
            recon_image, mean, log_var = model(image)
            z = self.reparameterize(mean, log_var)
        return recon_image.squeeze(0).cpu(), z.squeeze(0).cpu()  # Remove batch dimension
    
    def get_embedding(self, input_data, device="cuda"):
        self.eval()
        with torch.no_grad():
            input_data = input_data.to(device)
            if len(input_data.shape) == 3: 
                input_data = input_data.permute(2, 0, 1).unsqueeze(0)
            else:
                input_data = input_data.permute(0, 3, 1, 2)
            mean, log_var = self.encode(input_data)
            z = self.reparameterize(mean, log_var)
        return z.cpu()
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
    
    def save_model(self, model, config, path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(path, timestamp)
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")

        config_path = os.path.join(save_dir, "training_params.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Config saved at {config_path}")
        return save_dir
    
class CLVAE(VAE):
    def __init__(self, img_channels=1, latent_dim=128, img_height=16, img_width=16, projection_dim=64):
        super().__init__(img_channels, latent_dim, img_height, img_width)
        self.projection_network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, projection_dim)
        )

    def project(self, z):
        """
        Pass latent representations through the projection network.
        """
        return self.projection_network(z)
   
    def encode_image(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return z
    
    def decode_image(self, z):  
        return self.decode(z)
    
    def calculate_ContrastiveLoss(self, emb_i, emb_j, temperature=0.5, verbose=False):
        self.batch_size = emb_i.size(0)
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
            
        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
                
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size, )).to(emb_i.device).scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
            if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)
            
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )    
            if self.verbose: print("Denominator", denominator)
                
            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
                
            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss

    def calculate_loss(self, pos_pairs_batch, alpha=0.8, beta=0.2, temperature=0.1, rec_loss_type='mse', log_params=None):
        """
        Combined loss: Reconstruction loss + KL divergence + Contrastive loss.
        """
        
        pair_i, pair_j = pos_pairs_batch[0], pos_pairs_batch[1]

        if alpha != 0:
            recon_pair_i, mean_i, log_var_i = super().forward(pair_i)
            recon_pair_j, mean_j, log_var_j = super().forward(pair_j)

            loss_i = super().calculate_rec_loss(recon_pair_i, pair_i, mean_i, log_var_i, alpha=alpha, loss_type=rec_loss_type)
            loss_j = super().calculate_rec_loss(recon_pair_j, pair_j, mean_j, log_var_j, alpha=alpha, loss_type=rec_loss_type)

            if log_params is not None:
                # log the original and reconstructed images
                writer = log_params['writer']
                concat_pair_i = torch.cat((pair_i, recon_pair_i), dim=-1)
                grid_pair_i = torchvision.utils.make_grid(concat_pair_i, nrow=int(len(concat_pair_i)**0.5), pad_value=1, padding=8)
                writer.add_image(f"Original/Reconstructed Pair I {log_params['batch_idx']}", grid_pair_i, log_params['epoch'])

            recon_kl_loss = (loss_i + loss_j) / 2
        else:
            recon_kl_loss = torch.tensor(0.0)

        emb_i = self.project(self.encode_image(pair_i))
        emb_j = self.project(self.encode_image(pair_j))

        contrastive_loss = self.calculate_ContrastiveLoss(emb_i, emb_j, temperature=temperature)/len(pos_pairs_batch)

        total_loss = alpha * recon_kl_loss + beta * contrastive_loss

        return total_loss, recon_kl_loss, contrastive_loss

    def train_step(self, model, batch, optimizer, device="cuda", alpha=0.8, beta=0.2, temperature=0.1, rec_loss_type='mse'):
        """
        Perform a single training step with contrastive loss.
        """
        model.train()

        # Move data to device
        batch = batch.to(device).squeeze(0)        
        optimizer.zero_grad()

        loss, rec_loss, cl_loss = self.calculate_loss(batch, alpha=alpha, beta=beta, temperature=temperature, rec_loss_type=rec_loss_type)

        loss.backward()
        optimizer.step()

        return loss.item(), rec_loss.item(), cl_loss.item()

    def val_step(self, model, batch, device="cuda", alpha=0.8, beta=0.2, temperature=0.1, rec_loss_type='mse', log_params=None):
        """
        Perform a validation step with contrastive loss.
        """
        model.eval()

        with torch.no_grad():
            pos_pairs = batch.to(device).squeeze(0) 
            loss, rec_loss, cl_loss = self.calculate_loss(pos_pairs, alpha=alpha, beta=beta, temperature=temperature, rec_loss_type=rec_loss_type, log_params=log_params)

        return loss.item(), rec_loss.item(), cl_loss.item()
