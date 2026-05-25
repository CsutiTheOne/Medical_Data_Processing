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



# Carefully selected examples to see evidence for each in gradcam
EXAMPLES = ["Data/skin_lesions_classification/test/Melanoma/ISIC_0071342.jpg",
"Data/skin_lesions_classification/test/Melanoma/ISIC_0071904.jpg",
"Data/skin_lesions_classification/test/Melanoma/ISIC_0072001.jpg",
"Data/skin_lesions_classification/test/Melanoma/ISIC_0072101.jpg"]

# EXAMPLES = [
#     #Coatnet
#     #Melanoma as Melanoma
#     "Data/skin_lesions_classification/test/Melanoma/ISIC_0071904.jpg",
#     #Melanoma as Nevi
#     "Data/skin_lesions_classification/test/Melanoma/ISIC_0069583.jpg",
#     #Nevi as Melanoma
#     "Data/skin_lesions_classification/test/Melanocytic nevi/ISIC_0071585.jpg",
#     #Nevi as Nevi
#     "Data/skin_lesions_classification/test/Melanocytic nevi/ISIC_0071467.jpg",
#     #Swin
#     #Melanoma as Melanoma
#     "Data/skin_lesions_classification/test/Melanoma/ISIC_0071342.jpg",
#     #Melanoma as Nevi
#     #"Data/skin_lesions_classification/test/Melanoma/ISIC_0071904.jpg",
#     #Nevi as Melanoma
#     "Data/skin_lesions_classification/test/Melanocytic nevi/ISIC_0069937.jpg",
#     #Nevi as Nevi
#     "Data/skin_lesions_classification/test/Melanocytic nevi/ISIC_0071585.jpg",
#     #Both
#     #Melanoma as Melanoma
#     "Data/skin_lesions_classification/test/Melanoma/ISIC_0072001.jpg",
#     #Melanoma as Nevi
#     "Data/skin_lesions_classification/test/Melanoma/ISIC_0072101.jpg",
#     #Nevi as Melanoma
#     "Data/skin_lesions_classification/test/Melanocytic nevi/ISIC_0071091.jpg",
#     #Nevi as Nevi
#     "Data/skin_lesions_classification/test/Melanocytic nevi/ISIC_0071630.jpg"
# ]




