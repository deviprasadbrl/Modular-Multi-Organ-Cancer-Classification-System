import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


def grad_cam(model, target_layer, input_tensor, class_idx=None):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)

    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()

    score = output[0, class_idx]
    score.backward()

    fh.remove()
    bh.remove()

    act = activations[0][0]
    grad = gradients[0][0] 

    weights = grad.mean(dim=(1, 2))
    cam = torch.zeros(act.shape[1:], device=act.device)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam.detach().cpu().numpy(), class_idx


def overlay_heatmap(heatmap, original_image_path, save_path="gradcam_output.jpg", alpha=0.5):
    img = Image.open(original_image_path).convert("RGB")
    img = np.array(img)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlaid = np.uint8(img * (1 - alpha) + heatmap * alpha)
    Image.fromarray(overlaid).save(save_path)
    print(f"Saved: {save_path}")

    return overlaid