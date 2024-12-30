from predict import TwoPersonPredictor

def main():
    # Initialize the predictor
    predictor = TwoPersonPredictor()
    
    # Run setup
    print("Setting up the model...")
    predictor.setup()
    
    # Use direct URL strings instead of Path objects
    image1_url = "https://t4.ftcdn.net/jpg/02/45/56/35/360_F_245563558_XH9Pe5LJI2kr7VQuzQKAjAbz9PAyejG1.jpg" #"./ai_face2.png" #"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl1.png"
    image2_url = "https://replicate.delivery/pbxt/KHU47j4Ad3rbq6TVxRuwFhyyX6HYmWrCSlUuVOM3q3ORKgVt/demo.png" #"./ai_face2.png" #"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl2.png"
    
    print("Running prediction for two persons...")
    outputs = predictor.predict(
        image1=image1_url,
        image2=image2_url,
        prompt="2 people having dinner in the cafe in the Paris. Bodies and faces towards the camera.",
        prompt2="",
        #negative_prompt="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        negative_prompt="bad hands, bad anatomy, ugly, deformed, (face asymmetry, eyes asymmetry, deformed eyes, deformed mouth, open mouth)",
        scale=0.7,
        num_outputs=1,
        num_inference_steps=40,
        seed=None,
        output_width=512,
        output_height=512,
        use_refiner=False,
        use_face_detailer=False,
        face_refinement_strength=0.4
    )
    
    print(f"Generated {len(outputs)} combined images:")
    for i, output_path in enumerate(outputs):
        print(f"Output {i+1}: {output_path}")

if __name__ == "__main__":
    main() 