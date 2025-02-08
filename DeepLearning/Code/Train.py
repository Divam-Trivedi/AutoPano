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
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from torch import optim
from Network.Network import *
import json
import torch
import numpy as np
import gc, time
from torchview import draw_graph
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Modules Imported")


def GenerateBatch(JsonPath, MiniBatchSize):
    """
    Inputs:
    JsonPath - Path to the JSON file containing images and H4pt
    MiniBatchSize - Number of samples per batch

    Outputs:
    I1Batch - Batch of images (Tensor of shape [MiniBatchSize, 128, 128, 6])
    CoordinatesBatch - Batch of H4pt coordinates (Tensor of shape [MiniBatchSize, 8])
    """
    I1Batch = []
    I2Batch = []
    CoordinatesBatch = []

    # Load JSON file
    with open(JsonPath, 'r') as f:
        dataset = json.load(f)

    # Randomly select MiniBatchSize samples
    RandIdxs = np.random.choice(len(dataset), MiniBatchSize, replace=False)

    for idx in tqdm(RandIdxs):
        image = np.array(dataset[str(idx)]['stacked_input1'])  # Shape (128, 128, 6)
        imga = np.array(dataset[str(idx)]['stacked_input1'])[:, :, :3].astype(np.uint8)
        imgb = np.array(dataset[str(idx)]['stacked_input1'])[:, :, 3:].astype(np.uint8)
        imga = cv2.resize(imga, (128, 128), interpolation=cv2.INTER_LINEAR)
        imgb = cv2.resize(imgb, (128, 128), interpolation=cv2.INTER_LINEAR)
        img1 = np.reshape(imga, (128, 128, 3))
        img2 = np.reshape(imgb, (128, 128, 3))
        # image = np.concatenate((imga, imgb), axis=-1)
        h4pt = np.array(dataset[str(idx)]['H4pt_input2'])  # Shape (8,)

        # Convert to PyTorch Tensors
        I1Batch.append(torch.tensor(img1, dtype=torch.float32))
        I2Batch.append(torch.tensor(img2, dtype=torch.float32))
        CoordinatesBatch.append(torch.tensor(h4pt, dtype=torch.float32))

    return torch.stack(I1Batch).to(device, non_blocking=True), torch.stack(I2Batch).to(device,
                                                                                       non_blocking=True), torch.stack(
        CoordinatesBatch).to(device, non_blocking=True)


def read_json_in_chunks(file_path, chunk_size):
    with open(file_path, 'r') as f:
        data = {}
        for _ in range(chunk_size):
            try:
                line = next(f).strip()
                if line.startswith('{'):
                    obj = json.loads(line + '}')
                    data.update(obj)
            except StopIteration:
                break
            except json.JSONDecodeError:
                continue
        return data


def UnSupGenerateBatch(JsonPath, MiniBatchSize):
    """Memory-efficient batch generator with device-agnostic handling"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_sample(img_stacked, corners):
        if img_stacked.dtype != np.uint8:
            img_stacked = (img_stacked * 255).astype(np.uint8)

        img1 = img_stacked[:, :, :3]
        img2 = img_stacked[:, :, 3:]

        if img1.shape[0] > 0 and img1.shape[1] > 0:
            img1 = cv2.resize(img1.astype(np.uint8), (128, 128))
            img2 = cv2.resize(img2.astype(np.uint8), (128, 128))
        else:
            raise ValueError(f"Invalid image shape: {img1.shape}")

        img1_tensor = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img2_tensor = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_stacked_tensor = torch.cat((img1_tensor, img2_tensor), dim=0)
        corners_tensor = torch.tensor(corners, dtype=torch.float32)

        return img_stacked_tensor, corners_tensor

    batch_images = []
    batch_corners = []

    with open(JsonPath, 'r') as f:
        dataset = json.load(f)
        total_samples = len(dataset)
        selected_indices = np.random.choice(total_samples, MiniBatchSize, replace=False)

        for idx in selected_indices:
            try:
                sample_data = dataset[str(idx)]
                img_stacked = np.array(sample_data['stacked_input1'])
                corners = np.array(sample_data['cornersA']).flatten().astype(np.float32)

                if img_stacked.size == 0 or corners.size == 0:
                    continue

                img_tensor, corners_tensor = process_sample(img_stacked, corners)
                batch_images.append(img_tensor)
                batch_corners.append(corners_tensor)

            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                continue

    if not batch_images:
        raise RuntimeError("No valid samples were processed")

    batch_images = torch.stack(batch_images)
    batch_corners = torch.stack(batch_corners)

    return batch_images.to(device), batch_corners.to(device)

# def UnSupGenerateBatch(JsonPath, MiniBatchSize):
#     """
#     Inputs:
#     JsonPath - Path to the JSON file containing images and corners
#     MiniBatchSize - Number of samples per batch
#
#     Outputs:
#     I1Batch - Batch of stacked images (Tensor of shape [MiniBatchSize, 6, 128, 128])
#     CoordinatesBatch - Batch of corner coordinates (Tensor of shape [MiniBatchSize, 8])
#     """
#     I1Batch = []
#     CoordinatesBatch = []
#
#     # Load JSON file
#     with open(JsonPath, 'r') as f:
#         dataset = json.load(f)
#
#     # Randomly select MiniBatchSize samples
#     RandIdxs = np.random.choice(len(dataset), MiniBatchSize, replace=False)
#
#     for idx in RandIdxs:
#         image = np.array(dataset[str(idx)]['stacked_input1'])  # Shape (H, W, 6)
#
#         # Extract img1 and img2 (RGB images)
#         img1 = image[:, :, :3].astype(np.uint8)
#         img2 = image[:, :, 3:].astype(np.uint8)
#
#         # Resize images to 128x128
#         img1 = cv2.resize(img1, (128, 128), interpolation=cv2.INTER_LINEAR)
#         img2 = cv2.resize(img2, (128, 128), interpolation=cv2.INTER_LINEAR)
#
#         # Normalize and convert to PyTorch Tensor [C, H, W]
#         img1_tensor = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1) / 255.0
#         img2_tensor = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1) / 255.0
#
#         # Concatenate along channel dimension (Shape: [6, 128, 128])
#         img_stacked = torch.cat((img1_tensor, img2_tensor), dim=0)
#
#         # Extract cornersA as ground truth (Instead of H4pt)
#         corners = np.array(dataset[str(idx)]['cornersA']).flatten().astype(np.float32)
#
#         I1Batch.append(img_stacked)
#         CoordinatesBatch.append(torch.tensor(corners, dtype=torch.float32))
#
#     return torch.stack(I1Batch).to(device, non_blocking=True), torch.stack(CoordinatesBatch).to(device, non_blocking=True)




def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def print_training_progress(epoch, num_epochs, batch_idx, num_batches, loss, val_loss=None):
    """Print training progress in a consistent format"""
    progress = f"[{epoch}/{num_epochs}][{batch_idx}/{num_batches}]"
    loss_info = f"Loss: {loss:.4f}"

    if val_loss is not None:
        loss_info += f" | Val Loss: {val_loss:.4f}"

    print(f"{progress} {loss_info}")


def print_epoch_summary(epoch, train_losses, val_losses):
    """Print summary statistics for an epoch"""
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else None

    print("\n" + "=" * 50)
    print(f"Epoch {epoch} Summary:")
    print(f"Average Training Loss: {avg_train_loss:.4f}")
    if avg_val_loss is not None:
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print("=" * 50 + "\n")

# def TrainOperation(
#     DirNamesTrain,
#     TrainCoordinates,
#     NumTrainSamples,
#     ImageSize,
#     NumEpochs,
#     MiniBatchSize,
#     SaveCheckPoint,
#     CheckPointPath,
#     DivTrain,
#     LatestFile,
#     BasePath,
#     LogsPath,
#     ModelType,
#     JsonPath
# ):
#     """
#     Inputs:
#     ImgPH is the Input Image placeholder
#     DirNamesTrain - Variable with Subfolder paths to train files
#     TrainCoordinates - Coordinates corresponding to Train/Test
#     NumTrainSamples - length(Train)
#     ImageSize - Size of the image
#     NumEpochs - Number of passes through the Train data
#     MiniBatchSize is the size of the MiniBatch
#     SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
#     CheckPointPath - Path to save checkpoints/model
#     DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
#     LatestFile - Latest checkpointfile to continue training
#     BasePath - Path to COCO folder without "/" at the end
#     LogsPath - Path to save Tensorboard Logs
#         ModelType - Supervised or Unsupervised Model
#     Outputs:
#     Saves Trained network in CheckPointPath and Logs to LogsPath
#     """
#     # Predict output with forward pass
#     #model = HomographyModel().to(device)
#
#     if ModelType == 'Sup':
#         model = HomographyModel().to(device)
#     elif ModelType == 'Unsup':
#         model = UnsupervisedHomographyModel().to(device)
#
#     result = None
#     latest_loss = None
#
#     ###############################################
#     # Fill your optimizer of choice here!
#     ###############################################
#     Optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     # Tensorboard
#     # Create a summary to monitor loss tensor
#     Writer = SummaryWriter(LogsPath)
#
#     if LatestFile is not None:
#         CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
#         # Extract only numbers from the name
#         StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
#         model.load_state_dict(CheckPoint["model_state_dict"])
#         print("Loaded latest checkpoint with the name " + LatestFile + "....")
#     else:
#         StartEpoch = 0
#         print("New model initialized....")
#
#     for Epochs in tqdm(range(StartEpoch, NumEpochs)):
#         print("Epoch " + str(Epochs))
#         NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
#         for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
#             torch.cuda.empty_cache() if torch.cuda.is_available() else None
#             gc.collect()
#             I1Batch, I2Batch, CoordinatesBatch = GenerateBatch(
#                 JsonPath, MiniBatchSize
#             )
#             print("Batch Generated")
#
#             # Predict output with forward pass
#             Batch = 0, I1Batch, I2Batch, CoordinatesBatch, 0
#             # Training step
#             if ModelType == "Sup":
#                 PredicatedCoordinatesBatch = model(I1Batch, I2Batch)
#                 LossThisBatch = LossFn(PredicatedCoordinatesBatch, 0, 0, CoordinatesBatch)
#                 result = {"loss": LossThisBatch, "val_loss": LossThisBatch}
#             else:
#                 result = model.training_step(Batch, PerEpochCounter)
#                 LossThisBatch = result["loss"]
#
#             Optimizer.zero_grad()
#             LossThisBatch.backward()
#             Optimizer.step()
#
#             # Validation step
#             if ModelType == "Sup":
#                 val_result = {"val_loss": LossFn(PredicatedCoordinatesBatch, 0, 0, CoordinatesBatch)}
#             else:
#                 val_result = model.validation_step(Batch)
#
#             latest_loss = val_result["val_loss"]
#
#
#             # Save checkpoint
#             if PerEpochCounter % SaveCheckPoint == 0:
#                 SaveName = CheckPointPath + str(Epochs) + "a" + str(PerEpochCounter) + "model.ckpt"
#                 torch.save({
#                     "epoch": Epochs,
#                     "model_state_dict": model.state_dict(),
#                     "optimizer_state_dict": Optimizer.state_dict(),
#                     "loss": latest_loss
#                 }, SaveName)
#                 print("\n" + SaveName + " Model Saved...")
#
#             # Tensorboard logging
#             Writer.add_scalar(
#                 "LossEveryIter",
#                 latest_loss,
#                 Epochs * NumIterationsPerEpoch + PerEpochCounter
#             )
#             Writer.flush()
#
#             # Explicit cleanup
#             if 'PredicatedCoordinatesBatch' in locals():
#                 del PredicatedCoordinatesBatch
#             del I1Batch, I2Batch, CoordinatesBatch, LossThisBatch
#             gc.collect()
#
#         # Save model after each epoch
#         SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
#         torch.save({
#             "epoch": Epochs,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": Optimizer.state_dict(),
#             "loss": latest_loss
#         }, SaveName)
#         print("\n" + SaveName + " Model Saved...")


def TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
        JsonPath
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    # model = HomographyModel().to(device)
    # if ModelType == 'Sup': continue
    #     #model = HomographyModel().to(device)
    # elif ModelType == 'Unsup':
    model = UnsupervisedHomographyModel().to(device)

    result = None
    latest_loss = None
    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Tensorboard

    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        print("Epoch " + str(Epochs))
        train_losses = []
        val_losses = []
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            # Generate batch
            if ModelType == "Sup":
                I1Batch, I2Batch, CoordinatesBatch = GenerateBatch(JsonPath, MiniBatchSize)
            else:
                I1Batch, CornersBatch = UnSupGenerateBatch(JsonPath, MiniBatchSize)
                I1Batch, CornersBatch = I1Batch.to(device), CornersBatch.to(device)

            # Predict output with forward pass
            #Batch = 0, I1Batch, I2Batch, CoordinatesBatch, 0

            # Training step
            if ModelType == "Sup":
                PredicatedCoordinatesBatch = model(I1Batch, I2Batch)
             #   LossThisBatch = LossFn(PredicatedCoordinatesBatch, 0, 0, CoordinatesBatch)
                result = {"loss": LossThisBatch, "val_loss": LossThisBatch}
            else:
                Batch = (I1Batch, CornersBatch)
                # print(f"Shape of img_a: {I1Batch[:, :3].shape}")  # Expected: (batch_size, 3, 128, 128)
                # print(f"Shape of img_b: {I1Batch[:, 3:].shape}")  # Expected: (batch_size, 3, 128, 128)
                # print(f"Shape of CornersBatch: {CornersBatch.shape}")  # Expected: (batch_size, 8)

                result = model.training_step(Batch, PerEpochCounter)
                LossThisBatch = result["loss"]
                train_losses.append(LossThisBatch.item())


            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Validation step
            if ModelType == "Sup":
                val_result = {"val_loss": LossFn(PredicatedCoordinatesBatch, 0, 0, CoordinatesBatch)}
            else:
                val_result = model.validation_step(Batch)

            latest_loss = val_result["val_loss"]
            val_loss = val_result["val_loss"]
            val_losses.append(val_loss.item())

            # Tensorboard logging
            Writer.add_scalar(
                "LossEveryIter",
                LossThisBatch,
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            Writer.add_scalar(
                "ValLossEveryIter",
                latest_loss,
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            Writer.flush()

            # Print progress every N batches (e.g., every 10 batches)
            if PerEpochCounter % 10 == 0:
                print_training_progress(Epochs, NumEpochs, PerEpochCounter,
                                     NumIterationsPerEpoch, LossThisBatch.item(),
                                     val_loss.item())

            # Save checkpoint
            if PerEpochCounter % SaveCheckPoint == 0:
                SaveName = (
                        CheckPointPath
                        + str(Epochs)
                        + "a"
                        + str(PerEpochCounter)
                        + "model.ckpt"
                )
                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": latest_loss,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            # Explicitly delete tensors to free memory
            del I1Batch, LossThisBatch
            gc.collect()

        # Print epoch summary
        print_epoch_summary(Epochs, train_losses, val_losses)
        # Save model every epoch

        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": latest_loss,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")


def plot_training_metrics(logs_path):


    # Initialize event accumulator
    ea = event_accumulator.EventAccumulator(logs_path)
    ea.Reload()

    # Extract loss values
    train_loss = []
    val_loss = []

    if 'LossEveryIter' in ea.Tags()['scalars']:
        train_events = ea.Scalars('LossEveryIter')
        train_loss = [event.value for event in train_events]

    if 'ValLossEveryIter' in ea.Tags()['scalars']:
        val_events = ea.Scalars('ValLossEveryIter')
        val_loss = [event.value for event in val_events]

    # Create plot
    plt.figure(figsize=(10, 5))

    if train_loss:
        plt.plot(train_loss, label='Training Loss')
    if val_loss:
        plt.plot(val_loss, label='Validation Loss')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig(os.path.join(logs_path, 'training_metrics.png'))
    plt.close()

def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p1\\YourDirectoryID_p1\\Phase2\\Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--JsonPath",
        default="D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p1\\YourDirectoryID_p1\\Phase2\\Data\\val__json.json",
        help=f"JSON File Path, Default: /home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data/Val_json.json"
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p1\\YourDirectoryID_p1\\Phase2\\Checkpoints\\Val\\",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=64,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=4,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/Val",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType
    JsonPath = Args.JsonPath

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
        JsonPath
    )

    # Plot training and validation metrics
    plot_training_metrics(LogsPath)


if __name__ == "__main__":
    model = UnsupervisedHomographyModel().to(device)
    t1 = time.time()
    main()
    t2 = time.time()
    print(t2 - t1)