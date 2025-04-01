import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from matplotlib import pyplot as plt
from torchvision import datasets
import yaml
import os
from pytorch_Lightning import CIFAR10CNN


def load_model(checkpoint_path, model_params):
    # Load the trained model from the checkpoint
    model = CIFAR10CNN.load_from_checkpoint(
        checkpoint_path, model_params=model_params)
    model.eval()  # Set the model to evaluation mode
    return model


def predict(image_tensor, model):
    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        # Get the predicted class index
        predicted_class = outputs.argmax(1).item()
    return predicted_class


if __name__ == "__main__":

    # Load the configuration file

    with open("config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    callback_params = config['callbacks']
    data_params = config['data']
    model_params = config['model']
    training_params = config['training']
    cifar10_classes = data_params["cifar10_classes"]

    # Path to the checkpoint file
    checkpoint_path = callback_params["checkpoint"]["dirpath"]

# Ensure directory exists before listing files
if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
    checkpoint_files = sorted(os.listdir(
        checkpoint_path))  # List and sort files

    if checkpoint_files:
        checkpoint_path = os.path.join(
            checkpoint_path, checkpoint_files[-1])  # Get latest file

    else:
        checkpoint_path = None  # No checkpoints exist

else:
    checkpoint_path = None  # Directory doesn't exist

print(f"Latest checkpoint: {checkpoint_path}")

# Load the trained model
model = load_model(checkpoint_path, model_params)

# transforms
test_transforms = transforms.Compose([
    eval(transform) for transform in data_params['test_transforms']])

test_dataset = datasets.CIFAR10(
    root=data_params['data_dir'],
    train=False, download=True,
    transform=test_transforms)

# Save a sample image
# Define mean and std used in Normalize()
mean = torch.tensor([0.5, 0.5, 0.5])
std = torch.tensor([0.5, 0.5, 0.5])

image_idxs = torch.randint(0, len(test_dataset), (10,))

test_img_path = os.path.join("./", 'test_images')
os.makedirs(test_img_path, exist_ok=True)

for idx in image_idxs:

    image, label = test_dataset[idx]
    # Unnormalize the image
    image = image * std[:, None, None] + \
        mean[:, None, None]  # Reverse normalization

    # Convert to NumPy and ensure values are in [0,1]
    image = image.permute(1, 2, 0).numpy().clip(0, 1)

    plt.imsave(f"{test_img_path}/sample_image_{idx}.jpg", image)
    # Path to the new image
    image_path = f"{test_img_path}/sample_image_{idx}.jpg"

    # Preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = test_transforms(image).unsqueeze(0)

    # Predict the class
    predicted_class = predict(image_tensor, model)

    predicted_class_name = cifar10_classes[predicted_class]
    print(f"Predicted Class: {predicted_class_name}")
