import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image
import os
from tqdm import tqdm
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
import numpy as np

# https://huggingface.co/friedrichor/stable-diffusion-2-1-realistic
#https://huggingface.co/collections/SG161222/realistic-vision-sd15-656daddd8a37acfa3f30cf53
#https://github.com/hhj1897/face_parsing
class FacePairGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device
        self.cache_dir = "./model-cache"
        self.text2img = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        ).to(device)
        
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        ).to(device)
        
        self.face_detector = RetinaFacePredictor(threshold=0.8, device=device,
                                        model=(RetinaFacePredictor.get_model('mobilenet0.25')))
        
        self.face_parser = RTNetPredictor(
            device=device, ckpt=None, encoder='rtnet50', decoder='fcn', num_classes=14)

   
        # Enable attention slicing for lower memory usage
        self.text2img.enable_attention_slicing()
        self.inpaint.enable_attention_slicing()


    def generate_base_face(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            
        prompt = "RAW photo, a close up portrait photo of man (without beard:2), (clean shaved face:1.2), (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
        negative_prompt = "(beard, facial hair, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        
        image = self.text2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        
        return image

    def add_beard(self, base_image, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            
        prompt = "RAW photo, a close up portrait photo of man with a full beard, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
        negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        
        # Detect faces
        faces = self.face_detector(np.array(base_image), rgb=False)
        if len(faces) == 0:
            print("No face detected")
            return base_image

        # Get face parsing masks
        masks = self.face_parser.predict_img(np.array(base_image), faces, rgb=False)
        
        # Create beard mask using skin and nose information
        skin_mask = (masks == 1).astype(np.uint8)  # Index 1 for skin
        nose_mask = (masks == 6).astype(np.uint8)  # Index 6 for nose
        
        # Find the lowest point of the nose
        nose_indices = np.where(nose_mask > 0)
        if len(nose_indices[0]) == 0:
            return base_image
        nose_bottom = np.max(nose_indices[1])  # Get maximum y-coordinate
        
        # Create beard mask from skin below nose
        beard_mask = np.zeros_like(skin_mask)
        beard_mask[:, nose_bottom:, :] = skin_mask[:, nose_bottom:, :]
        
        # Ensure mask is 2D and has correct dimensions
        if len(beard_mask.shape) == 3:
            beard_mask = beard_mask.squeeze()  # Remove single-dimensional entries
        
        # Convert to PIL Image
        mask_image = Image.fromarray(beard_mask * 255)  # Convert to 0-255 range
        
        # Perform inpainting
        bearded_image = self.inpaint(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_image,
            mask_image=mask_image,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        
        return bearded_image  # Return after processing first face
        
        return base_image  # Return original if no valid faces found

    def generate_pair(self, output_dir, index, seed=None):
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate base face
        base_image = self.generate_base_face(seed)
        base_path = os.path.join(output_dir, f"clean_{index}.png")
        base_image.save(base_path)
        
        # Generate bearded version
        bearded_image = self.add_beard(base_image, seed)
        beard_path = os.path.join(output_dir, f"beard_{index}.png")
        bearded_image.save(beard_path)
        
        return base_image, bearded_image

def generate_dataset(num_pairs=100, output_dir="face_pairs", start_seed=42):
    generator = FacePairGenerator(model_id="SG161222/Realistic_Vision_V2.0")
    
    for i in tqdm(range(num_pairs)):
        seed = start_seed + i
        generator.generate_pair(output_dir, i, seed)

# Preview function to display some results
def preview_pairs(output_dir, num_preview=10):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(num_preview, 2, figsize=(10, 5*num_preview))
    
    for i in range(num_preview):
        clean_img = Image.open(os.path.join(output_dir, f"clean_{i}.png"))
        beard_img = Image.open(os.path.join(output_dir, f"beard_{i}.png"))
        
        axes[i, 0].imshow(clean_img)
        axes[i, 0].set_title(f"Clean {i}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(beard_img)
        axes[i, 1].set_title(f"Bearded {i}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Generate the dataset
if __name__ == "__main__":
    generate_dataset(num_pairs=100)
    preview_pairs("face_pairs", num_preview=10)