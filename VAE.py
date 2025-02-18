import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE_dataset(Dataset):
    def __init__(self, image_paths, target_colors, img_size=64):
        super(VAE_dataset).__init__()
        self.image_paths = image_paths
        self.target_colors = target_colors  
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Resize((img_size, img_size))
        ])

    def mask_extraction(self, image, color):
        # Convertiamo il colore target da BGR a HSV
        # Creiamo un array 1x1x3 con il colore target
        # color_bgr = np.uint8([[color]])
        # color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
        
        # Converti l'immagine in HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Per il rosso dobbiamo gestire il wrapping intorno a 180°
        if color == (0, 0, 255):  # Se è rosso (BGR)
            # Crea due maschere per il rosso dato che attraversa lo 0
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        
        elif color == (0, 255, 0):  # Se è verde (BGR)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv_image, lower_green, upper_green)
        
        #mask = cv2.GaussianBlur(mask, (9,9), 0)
        # Normalizza la maschera a valori tra 0 e 1
        mask = mask.astype(np.float32) / 255.0
        return mask
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)  # BGR format
        
        masks = []
        for color in self.target_colors:
            mask = self.mask_extraction(image, color)
            mask = self.transform(mask).squeeze(0)  # Remove channel dim after ToTensor
            masks.append(mask)
        return torch.stack(masks).to(torch.float32)  # Shape: (2, img_size, img_size)
    


class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim=6):
        super(VariationalAutoEncoder, self).__init__()
        self.device = device
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1 , out_channels=16 , kernel_size=3 , stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32 ,64 ,3 ,2, 1)
        self.conv4 = nn.Conv2d(64 ,128 ,3 ,2, 1)

        #Probabilistic Latent space
        self.fc_mean = nn.Linear(in_features=128*4*4, out_features=latent_dim)
        self.fc_cov = nn.Linear(128*4*4, latent_dim)


        #Decoder
        self.fc_dec = nn.Linear(latent_dim, 2048)
        self.convt1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.convt2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.convt3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.convt4 = nn.ConvTranspose2d(16, 1, 4, 2, 1)

        self.to(torch.float32)

    def encoder(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flat dimensions and channels
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)

        mean = self.fc_mean(x)
        log_std = self.fc_cov(x)

        return mean, log_std
    
    # Reparametrization is needed for backpropagation (make sampling differentiable)
    def reparametrize(self, mean, log_std):
        std = torch.exp(0.5 * log_std)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def decoder(self,z):
        z = self.fc_dec(z)
        #print(z.shape)
        z = z.view(z.size(0), 128, 4, 4)
        z = F.relu(self.convt1(z))
        z = F.relu(self.convt2(z))
        z = F.relu(self.convt3(z))
        z = F.sigmoid(self.convt4(z))
        return z

    def forward(self,x):
        mean, log_std = self.encoder(x)
        z = self.reparametrize(mean,log_std)
        reconst = self.decoder(z)
        return reconst, mean, log_std


# def visualize_masks(dataset, index):
#     masks = dataset[index]
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
#     axes[0].imshow(masks[0], cmap='gray')
#     axes[0].set_title('Green Mask')
#     axes[0].axis('off')
    
#     axes[1].imshow(masks[1], cmap='gray')
#     axes[1].set_title('Red Mask')
#     axes[1].axis('off')
    
#     plt.show()
        


# # Specifica la cartella da cui vuoi ottenere i file immagine
# folder_path = 'frames2'

# # Elenco di tutti i file .png nella cartella
# image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".png")]   
# target_colors = [(0, 255, 0), (0, 0, 255)]  # Green (BGR), Red (BGR)

# dataset = VAE_dataset(image_paths, target_colors)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
# visualize_masks(dataset, 0)

# # Training loop
# vae = VariationalAutoEncoder().to(device)
# optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4) 
# for epoch in range(100):
#     for batch in dataloader: #batch_idx, batch in enumerate(dataloader):
#         masks = batch.to(device).float()  # Shape: (batch_size, 2, 64, 64)
        
#         # Flatten the masks to feed into the VAE
#         for mask in masks:
#             recon_mask, mean, log_std = vae(mask.unsqueeze(1))  # Add channel dim (1, 64, 64)
            
#             # Compute loss
#             # Reconstruction loss
#             recon_loss = F.binary_cross_entropy(recon_mask, mask.unsqueeze(1), reduction='sum')
            
#             # KL divergence loss
#             kl_loss = -0.5 * torch.sum(1 + log_std - mean.pow(2) - log_std.exp())
            
#             # Total loss
#             loss = recon_loss + kl_loss
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#     print(f'Epoch [{epoch+1}/90], Total Loss: {loss.item():.4f}')

#     if (epoch + 1) % 10 == 0:
#         vae.eval()
#         with torch.no_grad():
#             sample_batch = next(iter(dataloader)).to(device)
#             # Prendi solo la prima maschera del batch (indice 0) e il primo canale (indice 0)
#             sample_mask = sample_batch[0, 0].unsqueeze(0).unsqueeze(0)  # Aggiungi dim batch e canale

#             recon_mask, _, _ = vae(sample_mask)

#             # Visualizza solo la prima maschera
#             recon_mask = recon_mask.squeeze().cpu().numpy()
#             sample_mask = sample_mask.squeeze().cpu().numpy()

#             fig, ax = plt.subplots(1, 2)
#             ax[0].imshow(sample_mask, cmap='gray')
#             ax[0].set_title('Original Mask (Green)')
#             ax[0].axis('off')

#             ax[1].imshow(recon_mask, cmap='gray')
#             ax[1].set_title('Reconstructed Mask (Green)')
#             ax[1].axis('off')

#             plt.show()

#             # Opzionale: visualizza anche la maschera del secondo colore
#             sample_mask_red = sample_batch[0, 1].unsqueeze(0).unsqueeze(0)  # Prendi la maschera rossa
#             recon_mask_red, _, _ = vae(sample_mask_red)

#             recon_mask_red = recon_mask_red.squeeze().cpu().numpy()
#             sample_mask_red = sample_mask_red.squeeze().cpu().numpy()

#             fig, ax = plt.subplots(1, 2)
#             ax[0].imshow(sample_mask_red, cmap='gray')
#             ax[0].set_title('Original Mask (Red)')
#             ax[0].axis('off')

#             ax[1].imshow(recon_mask_red, cmap='gray')
#             ax[1].set_title('Reconstructed Mask (Red)')
#             ax[1].axis('off')

#             plt.show()

#         vae.train()

# # Salva il modello alla fine dell'allenamento
# torch.save(vae.state_dict(), 'vae_model_new.pth')



