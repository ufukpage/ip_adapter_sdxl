import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from beard_removal_models import BeardRemovalVAE
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
import numpy as np

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

def calculate_metrics(pred_imgs, target_imgs, masks):
    psnr = PeakSignalNoiseRatio().to(pred_imgs.device)
    ssim = StructuralSimilarityIndexMeasure().to(pred_imgs.device)
    lpips_fn = lpips.LPIPS(net='alex').to(pred_imgs.device)

    # Apply masks to images before calculating metrics
    masked_pred = pred_imgs * masks
    masked_target = target_imgs * masks
    
    # Calculate metrics on masked images
    psnr_val = psnr(masked_pred, masked_target)
    ssim_val = ssim(masked_pred, masked_target)
    lpips_val = lpips_fn(masked_pred * 2 - 1, masked_target * 2 - 1).mean()
    
    return psnr_val.item(), ssim_val.item(), lpips_val.item()

def train(data_dir, num_epochs=100, batch_size=8, learning_rate=0.0002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Split dataset into train and validation
    full_dataset = BeardRemovalDataset(data_dir)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = BeardRemovalVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    best_psnr = 0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for beard_imgs, clean_imgs in pbar:  # Updated to include masks
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
        
        # Validation loop (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_psnr_list = []
            val_ssim_list = []
            val_lpips_list = []
            val_loss = 0
            
            with torch.no_grad():
                for beard_imgs, clean_imgs, masks in val_loader:  # Updated to include masks
                    beard_imgs = beard_imgs.to(device)
                    clean_imgs = clean_imgs.to(device)
                    masks = masks.to(device)
                    
                    recon_imgs, mu, logvar = model(beard_imgs)
                    loss, recon_loss, kld_loss = vae_loss(recon_imgs, clean_imgs, mu, logvar, masks)
                    
                    psnr, ssim, lpips_val = calculate_metrics(recon_imgs, clean_imgs, masks)
                    val_psnr_list.append(psnr)
                    val_ssim_list.append(ssim)
                    val_lpips_list.append(lpips_val)
                    val_loss += loss.item()
            
            avg_val_psnr = np.mean(val_psnr_list)
            avg_val_ssim = np.mean(val_ssim_list)
            avg_val_lpips = np.mean(val_lpips_list)
            avg_val_loss = val_loss / len(val_loader)
            
            print(f'\nValidation Metrics:')
            print(f'Loss: {avg_val_loss:.4f}')
            print(f'PSNR: {avg_val_psnr:.2f}')
            print(f'SSIM: {avg_val_ssim:.4f}')
            print(f'LPIPS: {avg_val_lpips:.4f}')
            
            # Save best model based on PSNR
            if avg_val_psnr > best_psnr:
                best_psnr = avg_val_psnr
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                    'psnr': avg_val_psnr,
                    'ssim': avg_val_ssim,
                    'lpips': avg_val_lpips
                }, 'best_model.pth')
        
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    print(f'\nBest model was saved at epoch {best_epoch+1} with PSNR: {best_psnr:.2f}')

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
    train("face_pairs2", num_epochs=300, batch_size=8)
    
    test_model(
        model_path="checkpoint_epoch_300.pth",
        test_image_path="face_pairs_test/beard_0.png",
        output_path="test_output.png"
    ) 