import os
import torch
import torchvision
import torchvision.transforms.functional as TF
import lpips

folder1_path = "style-images"
folder2_path = "output-images"
image1_path = os.path.join(folder1_path, "mosaic.jpg")
image2_path = os.path.join(folder2_path, "trained.jpg")  

# Loading the images
image1 = torchvision.io.read_image(image1_path).unsqueeze(0)
image2 = torchvision.io.read_image(image2_path).unsqueeze(0)

# Resizing the images to the same dimensions
image1 = TF.resize(image1, (256, 256))
image2 = TF.resize(image2, (256, 256))

# Creating the LPIPS model
loss_fn = lpips.LPIPS(net="vgg")

# Compute the LPIPS distance
distance = loss_fn(image1, image2)

print("LPIPS distance between the images:", distance.item())
