from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from diffusers.utils import load_image
import torch
from diffusers.image_processor import IPAdapterMaskProcessor
from PIL import Image
from insightface.app import FaceAnalysis
import cv2
import numpy as np

# Initialize face analysis with custom model directory
model_root = "./models/insightface/"  # Replace with your actual model directory path
app = FaceAnalysis(
    name="buffalo_l", 
    root=model_root,  # Specify custom model directory
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))

#"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl1.png"
image1_url = "https://t4.ftcdn.net/jpg/02/45/56/35/360_F_245563558_XH9Pe5LJI2kr7VQuzQKAjAbz9PAyejG1.jpg" #"./ai_face2.png" #"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl1.png"
#"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl2.png"
image2_url = "https://replicate.delivery/pbxt/KHU47j4Ad3rbq6TVxRuwFhyyX6HYmWrCSlUuVOM3q3ORKgVt/demo.png" 

cache_dir = "./model-cache"

# Initialize pipeline with specific model variant
"""
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")
"""

pipeline = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    device_map="balanced",
    low_cpu_mem_usage=True,
    cache_dir=cache_dir
)
# Load IP-Adapter with specific configuration

pipeline.load_ip_adapter("h94/IP-Adapter-FaceID",
                         subfolder=None, 
                         weight_name="ip-adapter-faceid_sdxl.bin", 
                         #image_encoder_folder="models/image_encoder"  # Add explicit image encoder path
                         )

# Set dimensions that match SDXL's ViT encoder expectations
output_height = 1024
output_width = 1024

# Load and process images
face_image1 = load_image(image1_url)
face_image2 = load_image(image2_url)

cv2_image = cv2.cvtColor(np.array(face_image1), cv2.COLOR_RGB2BGR)
faces = app.get(cv2_image)


face_image1 = face_image1.resize((output_width, output_height))
face_image2 = face_image2.resize((output_width, output_height))

# Load and process masks
mask1 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_mask1.png").convert('L')
mask2 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_mask2.png").convert('L')
mask1 = mask1.resize((output_width, output_height))
mask2 = mask2.resize((output_width, output_height))

# Process masks to get correct shape [1, num_images_for_ip_adapter, height, width]
processor = IPAdapterMaskProcessor()
masks = processor.preprocess([mask1, mask2], height=output_height, width=output_width)
masks = [masks.reshape(1, 2, output_height, output_width)]  # Reshape to correct dimensions

# Set IP adapter scale
pipeline.set_ip_adapter_scale(0.7)

generator = torch.Generator(device="cpu").manual_seed(0)

images = pipeline(
    prompt="2 people having dinner in the cafe in the Paris. Bodies and faces towards the camera.",
    ip_adapter_image=[[face_image1, face_image2]],
    negative_prompt="bad hands, bad anatomy, ugly, deformed, (face asymmetry, eyes asymmetry, deformed eyes, deformed mouth, open mouth)",
    num_inference_steps=40,
    num_images_per_prompt=1,
    generator=generator,
    cross_attention_kwargs={"ip_adapter_masks": masks}
)
image = images.images[0]

# Save the generated image
output_path = "generated_image.png"
image.save(output_path)
print(f"Image saved to {output_path}")
