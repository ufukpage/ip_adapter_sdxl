import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image
import os
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
import numpy as np

#https://huggingface.co/collections/SG161222/realistic-vision-sd15-656daddd8a37acfa3f30cf53
#https://github.com/hhj1897/face_parsing
class FacePairGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda", low_vram=False):
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

        if low_vram:
            self.text2img.enable_attention_slicing()
            self.inpaint.enable_attention_slicing()

        self.resolutions = [512, 768]

    def generate_base_face(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            
        # Randomly select resolution and use it for both width and height
        size = np.random.choice(self.resolutions)
        width = height = size
            
        # Define variations for age and ethnicity
        age_variations = [
            "young", "middle aged", "elderly",
            "in his 20s", "in his 30s", "in his 40s", "in his 50s", "in his 60s"
        ]
        ethnicity_variations = [
            "african", "caucasian", "middle eastern", 
            "hispanic", "south asian", "east asian"
        ]
        
        selected_age = np.random.choice(age_variations)
        selected_ethnicity = np.random.choice(ethnicity_variations)
            
        prompt = f"RAW photo, a close up headshot portrait photo of {selected_ethnicity} {selected_age} man (without beard:2), (clean shaved face:1.2), (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
        negative_prompt = "(beard, facial hair, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        
        image = self.text2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=9.0,
            scheduler_name="DPM++ 2M Karras",
            width=width,
            height=height
        ).images[0]
        
        return image
    
    def get_beard_mask(self, base_image):
        faces = self.face_detector(np.array(base_image), rgb=False)
        if len(faces) == 0:
            print("No face detected")
            return base_image

        
        masks = self.face_parser.predict_img(np.array(base_image), faces, rgb=False)
        
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
        
        if len(beard_mask.shape) == 3:
            beard_mask = beard_mask.squeeze()  
        
        final_mask = (masks > 0).astype(np.uint8)
        if len(final_mask.shape) == 3:
            final_mask = final_mask.squeeze()  
        return beard_mask, final_mask

    def add_beard(self, base_image, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            
        prompt = "RAW photo, a close up portrait photo of man with a well-groomed full beard, (natural dense beard:1.3), (realistic beard texture:1.2), (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
        negative_prompt = "(patchy beard:1.4), (sparse facial hair:1.3), deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs"
        
        try:
            beard_mask, full_mask = self.get_beard_mask(base_image)
        except Exception as e:
            raise Exception(e)
        
        mask_image = Image.fromarray(beard_mask * 255)
        full_mask_image = Image.fromarray(full_mask * 255)
        width, height = base_image.size
        
        bearded_image = self.inpaint(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_image,
            mask_image=mask_image,
            num_inference_steps=40,
            guidance_scale=8.0,
            scheduler_name="DPM++ 2M Karras",
            width=width,
            height=height
        ).images[0]
        
        return bearded_image, mask_image, full_mask_image
        

    def generate_pair(self, output_dir, index, seed=None):
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate base face
        base_image = self.generate_base_face(seed)
        base_path = os.path.join(output_dir, f"clean_{index}.png")
        base_image.save(base_path)
        
        # Generate bearded version
        try:
            bearded_image, mask_image, full_mask_image = self.add_beard(base_image, seed)
        except Exception as e:
            raise Exception(e)
        beard_path = os.path.join(output_dir, f"beard_{index}.png")
        bearded_image.save(beard_path)
        mask_path = os.path.join(output_dir, f"mask_{index}.png")
        mask_image.save(mask_path)
        full_mask_path = os.path.join(output_dir, f"full_mask_{index}.png")
        full_mask_image.save(full_mask_path)
        return base_image, bearded_image, mask_image, full_mask_image

def generate_dataset(num_pairs=100, output_dir="face_pairs", seed_range=1000000000):
    generator = FacePairGenerator(model_id="SG161222/Realistic_Vision_V2.0")
    
    seeds = np.random.randint(0, seed_range, size=num_pairs)
    
    i = 0
    while i<num_pairs:
        try:
            generator.generate_pair(output_dir, i, seeds[i])
            i += 1
        except Exception as e:
            print(e)

def preview_pairs(output_dir, num_preview=10):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(num_preview, 2, figsize=(10, 5*num_preview))
    # Randomize indices for preview
    available_indices = [i for i in range(len(os.listdir(output_dir))//4)]  # //4 since we have clean, beard, mask and full_mask
    preview_indices = np.random.choice(available_indices, size=min(num_preview, len(available_indices)), replace=False)
    for i in preview_indices:
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

if __name__ == "__main__":
    #print("Waiting for 1 hour before starting dataset generation...")
    #time.sleep(3600)  # Sleep for 3600 seconds (1 hour)
    generate_dataset(output_dir="face_pairs2", num_pairs=10000)
    preview_pairs("face_pairs", num_preview=10)