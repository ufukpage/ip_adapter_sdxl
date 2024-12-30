from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import torch
from diffusers.image_processor import IPAdapterMaskProcessor
from PIL import Image

# Initialize pipeline with specific model variant
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# Load IP-Adapter with specific configuration
pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
    image_encoder_folder="models/image_encoder"  # Add explicit image encoder path
)

# Set dimensions that match SDXL's ViT encoder expectations
output_height = 1024
output_width = 1024

# Load and process images
face_image1 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl1.png")
face_image2 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl2.png")
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
    prompt="2 girls",
    ip_adapter_image=[[face_image1, face_image2]],
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=20,
    num_images_per_prompt=1,
    generator=generator,
    cross_attention_kwargs={"ip_adapter_masks": masks}
)
image = images.images[0]

# Save the generated image
output_path = "generated_image.png"
image.save(output_path)
print(f"Image saved to {output_path}")
