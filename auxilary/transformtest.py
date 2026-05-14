import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch import tensor
import helpers as h
import random
import os

def plotImage(imgPath):
    img = Image.open(imgPath).convert("RGB")

    train_transform = h.ComposeTrainingTransforms()
    val_transform = h.ComposeRegularTransforms()

    for_train = train_transform(img)
    for_val = val_transform(img)

    # Tensor [C, H, W] -> image [H, W, C]
    preview_train = for_train.permute(1, 2, 0)
    preview_val = for_val.permute(1, 2, 0)

    mean = tensor(h.STD)
    std = tensor(h.MEAN)

    #unnormalize
    #preview_train = preview_train * mean + std
    #preview_val = preview_val * mean + std

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(preview_train)
    plt.title("For training")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(preview_val)
    plt.title("For validation")
    plt.axis("off")

    plt.show()