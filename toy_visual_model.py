import torch
from pretraining import CNNModule
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by your CNN
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet norms
])

# Create an instance of the model
model = CNNModule()

# Load the saved model weights
saved_weights_path = "visual_model_weights.pth"
saved_weights = torch.load(saved_weights_path)
model_dict = model.state_dict()

# Filter out the classifier weights
filtered_weights = {k: v for k, v in saved_weights.items() if k in model_dict}

# Load the filtered weights into the model
model.load_state_dict(filtered_weights)

# model.load_state_dict(torch.load(saved_weights_path))
model.eval()  # Set the model to evaluation mode