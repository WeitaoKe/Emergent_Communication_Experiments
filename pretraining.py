import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
import random
from PIL import Image
d_R = 0.55
class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()

        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Dropout(d_R),

            # Second convolutional layer
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Dropout(d_R),

            # Third convolutional layer
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Dropout(d_R),

            # Fourth convolutional layer
            # nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=2, padding=0),
            # nn.BatchNorm2d(20),
            # nn.ReLU(inplace=True),
            # nn.Dropout(d_R),

            # Fifth convolutional layer
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Dropout(d_R)
        )

        # To compute the size of the input for the fully connected layer,
        dummy_input = torch.randn(1, 3, 224, 224)
        output_size = self.conv_layers(dummy_input).view(1, -1).size(1)

        # Define the fully connected layer
        self.fc_layer = nn.Sequential(
            nn.Linear(output_size, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(d_R)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.contiguous().view(x.size(0), -1)
        features = self.fc_layer(x)
        return features


# Utility function to load and convert an image to a tensor using PIL.
def load_image_as_tensor(image_path, image_transform):
    """Load an image from a given path and apply the specified transforms."""
    image = Image.open(image_path)
    return image_transform(image)


def get_batch(data_transform, batch_size=96, train=True):
    """Generate a batch of images and labels using logic from simulate_interaction_batch.

    Args:
    - data_transform: torchvision.transforms object to preprocess the image data.
    - batch_size: Number of images to fetch in one batch.
    - train: Boolean indicating if fetching for training or validation.

    Returns:
    - A tuple containing batch of image tensors and corresponding labels.
    """

    # Choose directory based on training or validation
    image_directory = './ImagesGenerated/for_training' if train else './ImagesGenerated/for_testing'

    labels = [label for label in os.listdir(image_directory) if os.path.isdir(os.path.join(image_directory, label))]

    image_tensors_list = []
    label_list = []

    for _ in range(batch_size):
        target_label = random.choice(labels)
        label_idx = labels.index(target_label)
        label_list.append(label_idx)

        target_label_directory = os.path.join(image_directory, target_label)
        target_label_images = os.listdir(target_label_directory)
        target_image_filename = random.choice(target_label_images)
        target_image_path = os.path.join(target_label_directory, target_image_filename)

        # Convert image to tensor and append to list
        image_tensor = load_image_as_tensor(target_image_path, data_transform)
        image_tensors_list.append(image_tensor)

    return torch.stack(image_tensors_list, axis=0), torch.tensor(label_list, dtype=torch.long)


def main():
    # Create an instance of the model
    model = CNNModule()

    # Create an independent classifier layer based on the number of classes (labels)
    num_classes = len(
        next(os.walk('./ImagesGenerated/for_training'))[1])
    classifier = nn.Linear(2048, num_classes)

    # Define data preprocessing transforms for images
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Define parameters for training
    batch_size = 96
    """
    # Compute the total number of images for training and validation
    num_training_images = sum([len(files) for r, d, files in os.walk(
        '/Users/keweitao/PycharmProjects/LanguageEvolutionSimulation/ImagesGenerated/for_training')])
    num_val_images = sum([len(files) for r, d, files in os.walk(
        '/Users/keweitao/PycharmProjects/LanguageEvolutionSimulation/ImagesGenerated/for_testing')])

    # Compute the number of batches for training and validation
    num_training_batches = num_training_images // batch_size
    num_val_batches = num_val_images // batch_size"""
    num_training_batches = 70
    num_val_batches = 70

    # Loss function and optimizer definition
    parameters = list(model.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(parameters, lr=0.00020, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        classifier.train()
        total_train_loss = 0.0
        correct_train_predictions = 0  # Added for accuracy tracking

        for _ in range(num_training_batches):
            inputs, labels = get_batch(data_transform, batch_size, train=True)
            optimizer.zero_grad()
            features = model(inputs)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # Calculate the number of correct predictions
            _, predicted = outputs.max(1)
            correct_train_predictions += (predicted == labels).sum().item()

        # Compute the training accuracy
        train_accuracy = correct_train_predictions / (batch_size * num_training_batches)

        # Validation phase
        model.eval()
        classifier.eval()
        total_val_loss = 0.0
        correct_val_predictions = 0  # Added for accuracy tracking

        with torch.no_grad():
            for _ in range(num_val_batches):
                inputs, labels = get_batch(data_transform, batch_size, train=False)
                features = model(inputs)
                outputs = classifier(features)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                # Calculate the number of correct predictions for validation
                _, predicted = outputs.max(1)
                correct_val_predictions += (predicted == labels).sum().item()

        # Compute the validation accuracy
        val_accuracy = correct_val_predictions / (batch_size * num_val_batches)

        # Logging the training and validation loss & accuracy for the current epoch
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_train_loss / num_training_batches}, Training Accuracy: {train_accuracy:.2f}, Validation Loss: {total_val_loss / num_val_batches}, Validation Accuracy: {val_accuracy:.2f}")

    # Save ONLY the model weights (not including the classifier layer's weights)
    save_path = "visual_model_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


if __name__ == '__main__':
    main()
