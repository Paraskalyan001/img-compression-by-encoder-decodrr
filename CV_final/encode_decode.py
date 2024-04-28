import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


from torchsummary import summary


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),  # Adjusted kernel size and stride
            nn.Sigmoid()  # Output range [0, 1] for image pixels
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize the model with the same architecture
model = Autoencoder()

# Load the saved parameters into the model (map to CPU)
model.load_state_dict(torch.load('autoencoder_model_with_Aug.pth', map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()


# Path to the folder containing images
# folder_path = "/content/drive/MyDrive/cv project/images"

# Get a list of all image files in the folder

output_path = "outputs/"

def get_image_files(folder_path = "images/"):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(('.jpeg', '.jpg', '.png'))]
    return image_files

image_files = get_image_files()

# Define a function to encode and decode the image
def encode_decode_image(image_tensor):
        # Encode the image
        encoded_image = model.encoder(image_tensor)

        # Decode the encoded image
        decoded_image = model.decoder(encoded_image)

        # Convert the decoded image tensor back to numpy array
        decoded_image_np = decoded_image.squeeze(0).detach().numpy()

        return decoded_image_np



    # Function to display images
def display_images(original_image, decoded_image):
        plt.figure(figsize=(10, 5))

        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_image.squeeze(0).permute(1, 2, 0).cpu().numpy())  # Permute to correct dimensions
        plt.title('Original Image')
        plt.axis('off')

        # Plot decoded image
        plt.subplot(1, 2, 2)
        plt.imshow(decoded_image.squeeze(0).permute(1, 2, 0).cpu().numpy())  # Permute to correct dimensions
        plt.title('Decoded Image')
        plt.axis('off')

        plt.show()
# Open and process each image
for image_path in image_files:
    # Open the image using PIL
    image = Image.open(image_path)

    # Convert the image to a numpy array
    image_np = np.array(image)

    # Convert the image to a torch tensor and normalize it
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image_tensor = transform(image_np).unsqueeze(0)


    # Encode and decode the image
    encoded_decoded_image = encode_decode_image(image_tensor)
    
    print()

    # Display the original image and decoded image side by side
    display_images(image_tensor, torch.tensor(encoded_decoded_image))
