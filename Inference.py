import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from models_paths import brain_path, kidney_path, oral_path, router_path
from grad_cam import grad_cam, overlay_heatmap
from input_validation import validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

router_classes = ["brain", "kidney", "oral", "random"]
brain_classes = ["glioma", "meningioma", "notumor", "pituitary"]
kidney_classes = ["cyst", "normal", "stone", "tumor"]
oral_classes = ["normal", "oscc"]


def load_model(path, num_classes):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

router_model = load_model(router_path, len(router_classes))
brain_model = load_model(brain_path, len(brain_classes))
oral_model = load_model(oral_path, len(oral_classes))
kidney_model = load_model(kidney_path, len(kidney_classes))


def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        router_out = router_model(x)
        router_probs = torch.softmax(router_out, dim=1)[0]
        router_idx = torch.argmax(router_probs).item()
        router_class = router_classes[router_idx]
        router_confidence = router_probs[router_idx].item()

        if router_confidence < 0.5:
            print(f"Router has low confidence ({router_confidence:.2%}) - recommended manual review")
            exit(0)

        # step 2: send to the matching expert
        if router_class == "brain":
            expert_model = brain_model
            expert_classes = brain_classes

        elif router_class == "oral":
            expert_model = oral_model
            expert_classes = oral_classes

        elif router_class == "kidney":
            expert_model = kidney_model
            expert_classes = kidney_classes

        elif router_class == "random":
            print("The current model is only trained to process brain(mri),kidney(ct) and oral Histopathological")
            exit(0)

        expert_out = expert_model(x)
        expert_probs = torch.softmax(expert_out, dim=1)[0]

        top_idx = torch.argmax(expert_probs).item()
        top_confidence=expert_probs[top_idx].item()
 
        if top_confidence<0.5:
            print(f"Model has low confidence ({top_confidence:.2%}) - recommended manual review")
            exit(0)

        final_class=expert_classes[torch.argmax(expert_probs).item()]

    print(f"Router says: {router_class}")
    print(f"Final prediction: {final_class}")
    print("Confidence per class:")
    for cls, prob in zip(expert_classes, expert_probs.tolist()):
        print(f"  {cls}: {prob:.2%}")

    heatmap, _ = grad_cam(
        model=expert_model,
        target_layer=expert_model.features.denseblock4,
        input_tensor=x
    )
    save_path = f"{router_class}_gradcam.jpg"
    overlay_heatmap(heatmap, image_path, save_path=save_path)

    return router_class, final_class


if __name__ == "__main__":
    image_path="" #path of the image

    if(validate(image_path)):
        predict(image_path)


    else:
        print("The model only accepts png and jpg format")
