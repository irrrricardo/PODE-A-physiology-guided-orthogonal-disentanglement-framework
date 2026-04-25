import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch.nn.functional as F

# --- Import your model definitions and transforms ---
from ..model import AgePredictionViT
from ...shared.data_utils import get_transforms


def get_args():
    parser = argparse.ArgumentParser(description="Attention Rollout Visualization for ViT")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained .pth model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input fundus image')
    parser.add_argument('--output_dir', type=str, default='./rollout_results', help='Directory to save results')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='Model name')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--head_fusion', type=str, default='mean', choices=['mean', 'max', 'min'],
                        help='How to fuse attention heads')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='Ratio of pixels to discard (keep top 10%) for cleaner view')
    return parser.parse_args()


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean", discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []
        self.hooks = []

        print(f"DEBUG: Searching for layers named '{attention_layer_name}' to hook...")
        found_layers = 0

        # Iterate through all layers and try to register Hooks
        for name, module in self.model.named_modules():
            # Print layer names containing 'attn' to help us locate them
            if 'attn' in name and len(name.split('.')) < 4:  # Only print first few layers to avoid flooding
                print(f"  Found layer: {name} | Type: {type(module)}")

            if attention_layer_name in name:
                self.hooks.append(module.register_forward_hook(self.get_attention))
                found_layers += 1

        print(f"DEBUG: Successfully hooked {found_layers} Attention layers.")

        if found_layers == 0:
            print(
                "!!! Critical error: No layers hooked! Please check the layer name list above to see if 'attn_drop' was renamed (e.g., to 'drop' or something else).")

    def get_attention(self, module, input, output):
        # output is the result after dropout
        self.attentions.append(output.detach().cpu())

    def __call__(self, input_tensor):
        self.attentions = []

        # Use context manager to disable SDPA acceleration and force Python-level logic
        # enable_math=True allows using normal math operations (we will go this route)
        with torch.no_grad(), \
                torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            output = self.model(input_tensor)

        if len(self.attentions) == 0:
            raise RuntimeError("Forward pass completed, but self.attentions is still empty! This means the Hook didn't work.")

        return self.compute_rollout()

    def compute_rollout(self):
        # self.attentions is a list containing Attention Matrix for each layer
        # Shape of each: (Batch, Heads, N, N)
        # N = 1 (CLS) + H*W (Patches)

        # Initialize identity matrix (N, N)
        result = torch.eye(self.attentions[0].size(-1))

        # Move to CPU for computation to avoid running out of GPU memory
        result = result.cpu()

        for attention in self.attentions:
            # attention shape: (Batch, Heads, N, N)

            # 1. Fuse multi-heads (Multi-head Fusion)
            if self.head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif self.head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif self.head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                attention_heads_fused = attention.mean(axis=1)

            # Take the first sample from batch: (N, N)
            flat = attention_heads_fused[0].cpu()

            # 2. Simulate residual connection (Residual Connection)
            # Formula: 0.5 * Attention + 0.5 * Identity
            I = torch.eye(flat.size(-1))
            a = (flat + I) / 2

            # Re-normalize
            a = a / a.sum(dim=-1, keepdim=True)

            # 3. Recursive matrix multiplication
            result = torch.matmul(a, result)

        # Extract CLS token's attention to all patches (result[0, 1:])
        mask = result[0, 1:]

        # 4. Reshape back to 2D image
        width = int(mask.size(0) ** 0.5)
        mask = mask.reshape(width, width).numpy()

        # Normalize to [0, 1]
        mask = mask / np.max(mask)

        return mask

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()


def show_mask_on_image(img, mask):
    """
    Overlay heatmap on original image (optimized transparency version)
    img: float32, (H, W, 3), [0, 1] (RGB)
    mask: float32, (H, W), [0, 1]
    """
    # 1. Generate heatmap (cv2 defaults to BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    # 2. Convert BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0

    # ==================== 【Key modification: adjust transparency】 ====================
    # Previous logic: 0.6 * heatmap + 0.4 * original (heatmap too thick, covers original)
    # Modified logic: 0.3 * heatmap + 0.7 * original (let original show through)

    # Option A: Standard blending (sum to 1.0)
    # cam = 0.3 * heatmap + 0.7 * img

    # Option B (recommended): Keep original at 100% brightness, only overlay faint heatmap
    # This makes blood vessels clearest and background won't darken
    cam = 0.5 * heatmap + 0.8 * img
    # =================================================================

    # 3. Re-normalize to prevent overflow (because 0.4 + 1.0 could exceed 1.0)
    cam = cam / np.max(cam)

    return np.uint8(255 * cam)


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load model ---
    print(f"Loading model from {args.model_path}...")
    model = AgePredictionViT(
        model_name=args.model_name,
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        head_dropout_rate=0.0
    )
    checkpoint = torch.load(args.model_path, map_location=device)

    # Handle weights saved with DataParallel (if 'module.' prefix exists)
    if 'module.' in list(checkpoint.keys())[0]:
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # ==================== 【Key fix: Disable Fused Attention】 ====================
    # Reason: timm's Fused Attention will skip the attn_drop layer, preventing weight capture.
    # We need to force all Attention modules to use the standard computation path.
    print("Force disabling Fused Attention to capture weights...")
    for module in model.modules():
        # Check if it's a timm Attention module and has fused_attn attribute
        if hasattr(module, 'fused_attn'):
            module.fused_attn = False
    # ========================================================================

    # --- 2. Prepare image ---
    rgb_img = Image.open(args.image_path).convert('RGB')
    transform = get_transforms(image_size=args.image_size, is_train=False)
    input_tensor = transform(rgb_img).unsqueeze(0).to(device)

    # --- 3. Execute Attention Rollout ---
    # Initialize Rollout class
    rollout = VITAttentionRollout(model.vit, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio)

    # Get original mask (14x14)
    mask = rollout(input_tensor)

    # --- 4. Post-processing and visualization ---
    # ==================== 【Modification start: High-resolution redraw】 ====================

    # 1. Get original image's raw size (e.g., 2000x2000)
    orig_w, orig_h = rgb_img.size
    print(f"Original image size: {orig_w}x{orig_h}")

    # 2. Directly interpolate 14x14 mask to 【original size】, not 224
    # Use INTER_CUBIC interpolation to make heatmap edges smoother
    mask_high_res = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # 3. Create background removal mask (Binary Mask) at 【original size】
    # This ensures the mask edges are very sharp and fit blood vessels closely
    gray_img_high_res = np.array(rgb_img.convert('L'))
    _, binary_mask_high_res = cv2.threshold(gray_img_high_res, 10, 255, cv2.THRESH_BINARY)
    binary_mask_high_res = binary_mask_high_res.astype(np.float32) / 255.0

    # 4. Apply mask
    masked_mask_high_res = mask_high_res * binary_mask_high_res

    # 5. Prepare base image (convert PIL original to float32 numpy array)
    # This step preserves all HD details of the original image!
    img_float_high_res = np.float32(rgb_img) / 255.0

    # 6. Generate overlay (pass in HD original and HD mask)
    vis_img = show_mask_on_image(img_float_high_res, masked_mask_high_res)

    # ==================== 【Modification end】 ====================

    # --- 5. Save results ---
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.basename(args.image_path)
    name, ext = os.path.splitext(base_name)
    save_path = os.path.join(args.output_dir, f"{name}_rollout.jpg")

    # Convert to BGR for cv2 to save
    vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, vis_img_bgr)

    print(f"Rollout visualization saved to: {save_path}")

    # Print prediction value
    with torch.no_grad():
        pred_age = model(input_tensor).item()
        print(f"Predicted Age: {pred_age:.2f}")

    # Clean up hooks
    rollout.cleanup()


if __name__ == "__main__":
    main()