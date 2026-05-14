"""
    Constant values required for the training process.
"""

#Path to modified SLC Dataset from porject folder
DS_PATH = "Data/skin_lesions_classification"

#Pixelstats for normalization (derived by measurement)
MEAN = [0.62374597, 0.52011699, 0.50394945]
STD  = [0.24196318, 0.22335994, 0.23118716]

#Image size for model input
IMG_SIZE = (224, 224)   #A gpu-m nem bír többet :c
#Slightly larger size used before rotation
TILTED_SIZE = tuple(int(1.3*i) for i in IMG_SIZE)

