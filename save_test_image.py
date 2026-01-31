import torch
import torchvision
from PIL import Image
import os

def save_sample_image():
    # Load separate test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    
    # Get the first image and label
    image, label = testset[0]
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Saving image of class: {class_names[label]}")
    image.save('test_image.png')
    print("Saved to test_image.png")

if __name__ == "__main__":
    save_sample_image()
