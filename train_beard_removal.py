import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from beard_removal_models import BeardRemovalVAE
class BeardRemovalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Get all paired images
        self.beard_images = []
        self.clean_images = []
        
        for i in range(len(os.listdir(data_dir))//2):
            beard_path = os.path.join(data_dir, f"beard_{i}.png")
            clean_path = os.path.join(data_dir, f"clean_{i}.png")
            
            if os.path.exists(beard_path) and os.path.exists(clean_path):
                self.beard_images.append(beard_path)
                self.clean_images.append(clean_path)

    def __len__(self):
        return len(self.beard_images)

    def __getitem__(self, idx):
        beard_img = Image.open(self.beard_images[idx]).convert('RGB')
        clean_img = Image.open(self.clean_images[idx]).convert('RGB')
        
        if self.transform:
            beard_img = self.transform(beard_img)
            clean_img = self.transform(clean_img)
            
        return beard_img, clean_img


def vae_loss(recon_x, x, mu, logvar, kld_weight=0.0001):
    # Reconstruction loss (L1 + MSE)
    recon_loss = nn.L1Loss()(recon_x, x) + 0.1 * nn.MSELoss()(recon_x, x)
    
    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss = kld_loss * kld_weight
    
    return recon_loss + kld_loss, recon_loss, kld_loss

def train(data_dir, num_epochs=100, batch_size=8, learning_rate=0.0002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = BeardRemovalDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = BeardRemovalVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for beard_imgs, clean_imgs in pbar:
                beard_imgs = beard_imgs.to(device)
                clean_imgs = clean_imgs.to(device)
                
                optimizer.zero_grad()
                recon_imgs, mu, logvar = model(beard_imgs)
                
                loss, recon_loss, kld_loss = vae_loss(recon_imgs, clean_imgs, mu, logvar)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({
                    'loss': loss.item(),
                    'recon': recon_loss.item(),
                    'kld': kld_loss.item()
                })
        
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

def test_model(model_path, test_image_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BeardRemovalVAE().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    input_image = Image.open(test_image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output, _, _ = model(input_tensor)
    
    output_image = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
    output_image.save(output_path)

if __name__ == "__main__":
    train("face_pairs", num_epochs=300, batch_size=8)
    
    test_model(
        model_path="checkpoint_epoch_300.pth",
        test_image_path="face_pairs_test/beard_0.png",
        output_path="test_output.png"
    ) 