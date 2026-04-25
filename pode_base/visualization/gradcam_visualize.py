import argparse
import cv2
import numpy as np
import torch
from PIL import Image
import os

# --- Import from our custom modules ---
from ..model import AgePredictionViT
from ...shared.data_utils import get_transforms

# --- Import from grad-cam library ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform_vit(tensor, height=14, width=14):
    """
    A reshape function customized for ViT models.
    pytorch-grad-cam library needs it to convert Transformer output (B, N, C) back to (B, C, H, W) image format.
    N = 1 (CLS token) + H*W (patch tokens)
    """
    # Remove CLS token because it doesn't correspond to an image spatial position
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Convert dimension to (B, C, H, W)
    result = result.permute(0, 3, 1, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualization for age prediction ViT model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights (.pth file)')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image for visualization')
    parser.add_argument('--output_dir', type=str, default='./grad_cam_results', help='Target directory to save visualization results')
    # --- Model parameters, must be consistent with training ---
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='ViT model name in timm library')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')

    args = parser.parse_args()

    # --- 1. Device setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Load model ---
    print(f"Loading model from '{args.model_path}'...")
    # Initialize the exact same model structure as during training, but don't load pretrained weights (pretrained=False)
    # Because we will load all fine-tuned weights from the file
    model = AgePredictionViT(
        model_name=args.model_name,
        pretrained=False,
        # Dropout should be set to 0 during inference, but eval() mode handles it automatically
        # For safety, you can manually set it
        drop_rate=0.0,
        drop_path_rate=0.0,
        head_dropout_rate=0.0
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()  # !!! Extremely important: switch to evaluation mode !!!
    print("Model loaded.")

    # --- 3. Prepare input image ---
    # Use PIL to load, ensure it's RGB format
    rgb_img = Image.open(args.image_path).convert('RGB')

    # Get image transformation pipeline for validation/testing
    transform = get_transforms(image_size=args.image_size, is_train=False)
    input_tensor = transform(rgb_img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # For final visualization, we also need a numpy array in 0-1 range
    rgb_img_float = np.float32(rgb_img) / 255

    # --- 4. Setup Grad-CAM ---
    # Target layer: usually select the output of the last Transformer Block or the LayerNorm layer after it
    # You can run print(model) to check model structure and determine specific names
    target_layer = model.vit.blocks[-1].norm1

    # Create GradCAM instance
    # Note: We passed in the ViT-specific `reshape_transform`
    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=lambda x: reshape_transform_vit(x, height=args.image_size // 16, width=args.image_size // 16)
    )

    # Target: For regression task, we don't care about a specific class, but why the model outputs the current value
    # RawScoresOutputTarget is used for this purpose
    targets = [RawScoresOutputTarget()]

    # --- 5. Generate and save visualization results ---
    print("Generating Grad-CAM...")
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # ==================== 【Key modification: start】 ====================
    # Purpose: Create a mask that only keeps the heatmap inside the circular fundus region, removing background and edge artifacts

    # 1. Resize original PIL image to match heatmap size (224x224)
    resized_rgb_img = rgb_img.resize((args.image_size, args.image_size))

    # 2. Convert to grayscale for threshold processing
    resized_gray_img_np = np.array(resized_rgb_img.convert('L'))

    # 3. Create binary mask. We assume background is pure black (pixel value close to 0), fundus region is brighter.
    # The threshold value 10 here is an empirical value, effective for most black background fundus images.
    _, binary_mask = cv2.threshold(resized_gray_img_np, 10, 255, cv2.THRESH_BINARY)

    # 4. Normalize mask from (0, 255) to (0, 1) range for multiplication operation
    binary_mask = binary_mask.astype(np.float32) / 255.0

    # 5. Apply mask to Grad-CAM heatmap, background area heat will become 0
    masked_grayscale_cam = grayscale_cam * binary_mask

    # ==================== 【Key modification: end】 ====================

    # Resize the processed heatmap (masked_grayscale_cam) to match high-resolution original image
    resized_masked_cam = cv2.resize(masked_grayscale_cam, (rgb_img_float.shape[1], rgb_img_float.shape[0]))
    # Overlay the processed CAM (resized_masked_cam) on the original image to generate 'visualization' variable
    visualization = show_cam_on_image(rgb_img_float, resized_masked_cam, use_rgb=True)

    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    # --- Key modification: auto-generate filename and save ---
    # 1. Ensure target output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Extract base filename from input path (e.g., "test_image.jpg")
    base_name = os.path.basename(args.image_path)

    # 3. Separate filename and extension (e.g., "test_image", ".jpg")
    name, ext = os.path.splitext(base_name)

    # 4. Create new filename (e.g., "test_image_cam.jpg")
    new_filename = f"{name}_cam{ext}"

    # 5. Combine output directory and new filename to get final save path
    save_path = os.path.join(args.output_dir, new_filename)

    # 6. Save image
    cv2.imwrite(save_path, visualization_bgr)
    print(f"Grad-CAM visualization result saved to: {save_path}")

    # Extra print of model's predicted age
    with torch.no_grad():
        predicted_age = model(input_tensor).item()
        print(f"Model's predicted age for this image: {predicted_age:.2f}")


if __name__ == "__main__":
    main()