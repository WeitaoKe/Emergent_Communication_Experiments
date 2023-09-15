import torch
from pretraining import CNNModule
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
preprocess = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),  # Resize to the input size expected by your CNN
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet norms
])

# --- Model Initialization ---
model = CNNModule()

# Load the saved model weights
saved_weights_path = "visual_model_weights.pth"
model.load_state_dict(torch.load(saved_weights_path))

model.eval()


def process_image(image_batch_tensor):
    # Apply preprocessing directly to the tensor
    preprocessed_tensor = preprocess(image_batch_tensor)

    with torch.no_grad():
        features = model(preprocessed_tensor)
    # print(features)
    return features.cpu().numpy()


