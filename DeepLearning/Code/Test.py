#!/usr/bin/env python
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
from Network.Network import *
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as transforms
import json
# Define input size
MP, NP = 128, 128  # Model's expected input size

# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize


def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    Img = np.float32(Img) / 255.0
    return Img


def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    I1 = Img

    if I1 is None:
        # OpenCV returns empty list if image is not read!
        print("ERROR: Image I1 cannot be read")
        sys.exit()

    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1


# def TestOperation(ImageSize, ModelPath, LabelsPathPred, test_json):
#     """
#     Inputs:
#     ImageSize is the size of the image
#     ModelPath - Path to load trained model from
#     TestSet - The test dataset
#     LabelsPathPred - Path to save predictions
#     Outputs:
#     Predictions written to /content/data/TxtFiles/PredOut.txt
#     """
#     # Predict output with forward pass, MiniBatchSize for Test is 1
#     #model = CIFAR10Model(InputSize=3 * 32 * 32, OutputSize=10)
#
#
#     # Move to GPU if available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = UnsupervisedHomographyModel().to(device)
#
#     CheckPoint = torch.load(ModelPath,map_location=device)
#     model.load_state_dict(CheckPoint["model_state_dict"],strict=False)
#     model.eval()
#     print(
#         "Number of parameters in this model are %d " % len(model.state_dict().items())
#     )
#
#     OutSaveT = open(LabelsPathPred, "w")
#
#     # Define transformation for input image
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((MP, NP)),  # Resize to match model input
#     ])
#
#     # Load test data from JSON
#     with open(test_json, "r") as f:
#         test_data = json.load(f)  # Load JSON dictionary
#
#     total_loss = 0.0
#     num_samples = 0
#
#     # Iterate through test samples
#     for sample in range(len(test_data)):
#         stacked_input1 = np.array(test_data[str(sample)]["stacked_input1"])  # (M, N, 6)
#         H4pt_input2 = np.array(test_data[str(sample)]["H4pt_input2"])  # (4, 2)
#
#         # Convert stacked_input1 to uint8
#         stacked_input1 = stacked_input1.astype(np.uint8)  # Convert to uint8
#
#         # Resize stacked_input1 (M, N, 6) → (MP, NP, 6)
#         stacked_input1_resized = np.zeros((MP, NP, 6), dtype=np.uint8)
#         for i in range(6):  # Resize each channel separately
#             stacked_input1_resized[:, :, i] = np.array(
#                 Image.fromarray(stacked_input1[:, :, i]).resize((NP, MP), Image.BILINEAR)
#             )
#         # Split the resized stack into two images (img1, img2) of shape (MP, NP, 3)
#         img1_resized = stacked_input1_resized[:, :, :3]  # First 3 channels
#         img2_resized = stacked_input1_resized[:, :, 3:]  # Last 3 channels
#
#         # Convert to PyTorch tensors
#         img1_tensor = torch.tensor(img1_resized, dtype=torch.float32).unsqueeze(0).to(device)
#         img2_tensor = torch.tensor(img2_resized, dtype=torch.float32).unsqueeze(0).to(device)
#
#         # Flatten H4pt_input2 from (4, 2) → (8,)
#         H4pt_input2_tensor = torch.tensor(H4pt_input2.flatten(), dtype=torch.float32).to(device)
#
#         # Forward pass to predict homography
#         with torch.no_grad():
#             pred_H4Pt = model(img1_tensor, img2_tensor)
#
#         pred_H4Pt = torch.floor(pred_H4Pt)
#
#         # Compute L2 loss
#         loss = torch.norm(pred_H4Pt - H4pt_input2_tensor, p=2)
#         total_loss += loss.item()
#         num_samples += 1
#
#         print(f"Sample {num_samples}: Loss = {loss.item():.6f}")
#
#     # Compute and print average loss
#     avg_loss = total_loss / num_samples
#     print(f"\n✅ Average Test Loss: {avg_loss:.6f}")


    # for count in tqdm(range(len(TestSet))):
    #     Img, Label = TestSet[count]
    #     Img, ImgOrg = ReadImages(Img)
    #
    #     # Move to device
    #     Img = torch.from_numpy(Img).to(device)
    #     if Label is not None:
    #         Label = torch.from_numpy(Label).to(device)
    #
    #     # Create batch for validation step
    #     Batch = [Img, Img[:, :3], Img[:, 3:], None, Label]
    #
    #     with torch.no_grad():
    #         result = model.validation_step(Batch)
    #         val_loss = result["val_loss"]
    #
    #     #PredT = torch.argmax(model(Img)).item()####
    #
    #     #OutSaveT.write(str(PredT) + "\n") ####
    #     OutSaveT.write(str(val_loss.cpu().numpy()) + "\n")
    # OutSaveT.close()

def TestOperation(ImageSize, ModelPath, LabelsPathPred, test_json):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnsupervisedHomographyModel().to(device)

    CheckPoint = torch.load(ModelPath, map_location=device)
    model.load_state_dict(CheckPoint["model_state_dict"], strict=False)
    model.eval()

    total_loss = 0.0
    predictions = []
    ground_truths = []
    batch_times = []

    # Load test data
    with open(test_json, "r") as f:
        test_data = json.load(f)

    for sample in tqdm(range(len(test_data)), desc="Testing"):
        start_time = time.time()

        # Load and process image
        stacked_input = np.array(test_data[str(sample)]["stacked_input1"])
        h4pt_gt = np.array(test_data[str(sample)]["H4pt_input2"])

        # Process each channel
        resized_input = np.zeros((128, 128, 6), dtype=np.float32)
        for i in range(6):
            channel = stacked_input[:, :, i].astype(np.uint8)
            resized_channel = cv2.resize(channel, (128, 128), interpolation=cv2.INTER_LINEAR)
            resized_input[:, :, i] = resized_channel / 255.0  # Normalize here

        # Convert to tensor
        img_tensor = torch.from_numpy(resized_input).float().permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        h4pt_gt = torch.from_numpy(h4pt_gt.flatten()).float().to(device)

        # Forward pass
        with torch.no_grad():
            pred_h4pt = model(img_tensor)
            loss = F.l1_loss(pred_h4pt.squeeze(), h4pt_gt)

            total_loss += loss.item()
            predictions.append(pred_h4pt.cpu().numpy().squeeze())
            ground_truths.append(h4pt_gt.cpu().numpy())

        batch_times.append(time.time() - start_time)

    # Calculate metrics
    num_samples = len(test_data)
    avg_loss = total_loss / num_samples
    avg_time = np.mean(batch_times)

    print('\nTest Results:')
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Average Processing Time: {avg_time * 1000:.2f} ms per image')

    # Create plots
    plt.figure(figsize=(15, 5))

    # Loss per sample
    plt.subplot(131)
    sample_losses = [F.l1_loss(torch.tensor(p), torch.tensor(g)).item()
                     for p, g in zip(predictions, ground_truths)]
    plt.plot(range(num_samples), sample_losses)
    plt.title('Loss per Sample')
    plt.xlabel('Sample')
    plt.ylabel('L1 Loss')

    # Error histogram
    plt.subplot(132)
    error_distribution = [np.mean(np.abs(p - g))
                          for p, g in zip(predictions, ground_truths)]
    plt.hist(error_distribution, bins=30)
    plt.title('Error Distribution')
    plt.xlabel('Average Error')
    plt.ylabel('Count')

    # Processing time
    plt.subplot(133)
    plt.plot(range(num_samples), batch_times)
    plt.title('Processing Time')
    plt.xlabel('Sample')
    plt.ylabel('Time (s)')

    plt.tight_layout()
    plt.savefig('test_metrics.png')

    # Save predictions
    with open(LabelsPathPred, 'w') as f:
        for pred in predictions:
            f.write(' '.join(map(str, pred)) + '\n')

    return avg_loss
def Accuracy(Pred, GT):
    """
    Inputs:
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return np.sum(np.array(Pred) == np.array(GT)) * 100.0 / len(Pred)


def ReadLabels(LabelsPathTest, LabelsPathPred):
    if not (os.path.isfile(LabelsPathTest)):
        print("ERROR: Test Labels do not exist in " + LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, "r")
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if not (os.path.isfile(LabelsPathPred)):
        print("ERROR: Pred Labels do not exist in " + LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, "r")
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())

    return LabelTest, LabelPred


def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(
        y_true=LabelsTrue, y_pred=LabelsPred  # True class for test-set.
    )  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + " ({0})".format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print("Accuracy: " + str(Accuracy(LabelsPred, LabelsTrue)), "%")


def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="/home/chahatdeep/Downloads/Checkpoints/144model.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default="/home/chahatdeep/Downloads/aa/CMSC733HW0/CIFAR10/Test/",
        help="Path to load images from, Default:BasePath",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default="./TxtFiles/LabelsTest.txt",
        help="Path of labels file, Default:./TxtFiles/LabelsTest.txt",
    )
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath

    # Setup all needed parameters including file reading
    #ImageSize = SetupAll()
    ImageSize = None

    # Define PlaceHolder variables for Input and Predicted output
    #ImgPH = tf.placeholder("float", shape=(1, ImageSize[0], ImageSize[1], 3))
    LabelsPathPred = "./TxtFiles/PredOut.txt"  # Path to save predicted labels

    ModelPath = "D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p1\\YourDirectoryID_p1\\Phase2\\Checkpoints\\Train\\49model.ckpt"
    test_json = "D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p1\\YourDirectoryID_p1\\Phase2\\Data\\val__json.json"
    TestOperation(ImageSize, ModelPath, LabelsPathPred, test_json)

    # # Plot Confusion Matrix
    # LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    # ConfusionMatrix(LabelsTrue, LabelsPred)


if __name__ == "__main__":
    main()