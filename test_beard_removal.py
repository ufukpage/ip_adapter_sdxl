import torch
from torchvision import transforms
from PIL import Image
import os
from train_beard_removal import BeardRemovalVAE
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class BeardRemovalTester:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = BeardRemovalVAE().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Inverse transform for visualization
        self.inverse_transform = transforms.Compose([
            transforms.Lambda(lambda x: x * 0.5 + 0.5),
            transforms.ToPILImage()
        ])

    def process_image(self, image_path):
        """Process a single image and return the de-bearded version"""
        input_image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output, mu, logvar = self.model(input_tensor)
        
        output_image = self.inverse_transform(output[0].cpu())
        return output_image, input_image.resize((512, 512))

    def process_directory(self, input_dir, output_dir):
        """Process all images in a directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc="Processing images"):
            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, f"debearded_{img_file}")
            
            output_image, _ = self.process_image(input_path)
            output_image.save(output_path)

    def visualize_results(self, input_path, output_path=None, show=True):
        """Visualize the before/after results"""
        output_image, input_image = self.process_image(input_path)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Show input image
        plt.subplot(1, 2, 1)
        plt.imshow(input_image)
        plt.title('Input Image')
        plt.axis('off')
        
        # Show output image
        plt.subplot(1, 2, 2)
        plt.imshow(output_image)
        plt.title('De-bearded Image')
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path)
        
        if show:
            plt.show()
            
        plt.close()

    def batch_process_and_visualize(self, input_dir, output_dir, num_samples=5):
        """Process multiple images and create a grid visualization"""
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) > num_samples:
            image_files = np.random.choice(image_files, num_samples, replace=False)
        
        fig = plt.figure(figsize=(12, 3*len(image_files)))
        
        for idx, img_file in enumerate(image_files):
            input_path = os.path.join(input_dir, img_file)
            output_image, input_image = self.process_image(input_path)
            
            # Save individual result
            output_image.save(os.path.join(output_dir, f"debearded_{img_file}"))
            
            # Add to visualization
            plt.subplot(len(image_files), 2, 2*idx + 1)
            plt.imshow(input_image)
            plt.title(f'Input {idx+1}')
            plt.axis('off')
            
            plt.subplot(len(image_files), 2, 2*idx + 2)
            plt.imshow(output_image)
            plt.title(f'Output {idx+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'results_grid.png'))
        plt.close()

def main():
    # Initialize tester
    tester = BeardRemovalTester(
        model_path="checkpoint_epoch_200.pth"  # Update with your model path
    )
    
    if not os.path.exists("results"):
        os.makedirs("results")
    # Test single image
    tester.visualize_results(
        input_path="face_pairs_test/test1.jpeg",  # Update with your test image path
        output_path="results/single_result.png"
    )
    
    # Batch process directory
    tester.process_directory(
        input_dir="face_pairs_test",  # Update with your test directory
        output_dir="results/individual"
    )
    
    # Create visualization grid
    tester.batch_process_and_visualize(
        input_dir="face_pairs_test",  # Update with your test directory
        output_dir="results/grid",
        num_samples=5
    )

if __name__ == "__main__":
    main() 