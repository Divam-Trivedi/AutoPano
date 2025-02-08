"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code

Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""
import pylab as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import cv2
import kornia  # You can use this to get the transform and warp in this project


# Don't generate pyc codes
sys.dont_write_bytecode = True

# def warp_perspective_opencv(image, H, output_size):
#
#     batch_size, channels, height, width = image.shape
#     warped_images = []
#
#     # Loop through each image in the batch
#     for i in range(batch_size):
#         # Convert PyTorch tensor to NumPy array
#         img_np = image[i].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
#         H_np = H[i].cpu().numpy()  # (3, 3)
#
# #         # Warp the image using OpenCV
# #         warped_np = cv2.warpPerspective(
# #             src=img_np,
# #             M=H_np,
# #             dsize=output_size,  # (width, height)
# #             flags=cv2.INTER_LINEAR,
# #             borderMode=cv2.BORDER_CONSTANT,
# #             borderValue=0  # Fill with black
# #         )
#
#         # Convert back to PyTorch tensor
#         warped_tensor = torch.from_numpy(warped_np).permute(2, 0, 1)  # (C, H, W)
#         warped_images.append(warped_tensor)
#
#     # Stack the warped images into a batch
#     warped = torch.stack(warped_images, dim=0)  # (Batch, C, H, W)
#     return warped

def LossFn(delta, img_a, patch_b, corners):
    """
    L2 Loss for homography estimation.
    """
    H4Pt_tilde = delta.reshape(-1, 8)
    H4Pt = corners.reshape(-1, 8)
    loss = torch.norm(H4Pt_tilde - H4Pt, p=2)
    return loss



# class HomographyNet(nn.Module):
#     def __init__(self):
#         super(HomographyNet, self).__init__()
#
#         # Convolutional layers
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),  # (128x128x32)
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),  # (64x64x32)
#
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (64x64x64)
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),  # (32x32x64)
#
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (32x32x128)
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),  # (16x16x128)
#
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (16x16x256)
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),  # (8x8x256)
#         )
#
#         # Fully connected layers
#         self.fc_layers = nn.Sequential(
#             nn.Linear(8 * 8 * 256, 512),
#             nn.ReLU(),
#             nn.Linear(512, 8)  # Output: H4Pt (8 values)
#         )
#
#     def forward(self, xa, xb):
#         """
#         Forward pass for homography estimation.
#         - xa: Image A (Batch x 128x128x3)
#         - xb: Image B (Batch x 128x128x3)
#         Returns:
#         - Predicted H4Pt (Batch x 8)
#         """
#         # xa = np.array([xa])
#         # xb = np.array([xb])
#         # print(xa.shape, xb.shape)
#         assert xa.shape[1:] == (128, 128, 3), "Input images must have shape (128, 128, 3) but is {} and {}".format(np.shape(xa), np.shape(xb))
#         assert xb.shape[1:] == (128, 128, 3), "Input images must have shape (128, 128, 3) but is {} and {}".format(np.shape(xa), np.shape(xb))
#
#         # Concatenate along the channel dimension to form (Batch, 128, 128, 6)
#         input_tensor = torch.cat((xa, xb), dim=-1)  # Concatenate along channel axis
#         input_tensor = input_tensor.permute(0, 3, 1, 2).contiguous()  # Convert to (Batch, 6, 128, 128)
#
#         # Pass through convolutional layers
#         features = self.conv_layers(input_tensor)
#
#         # Flatten features
#         features = features.reshape(features.shape[0], -1)
#
#         # Predict homography
#         out = self.fc_layers(features)
#         return out
#
#
# class HomographyModel(pl.LLightningModule):
#     def __init__(self, hparams):
#         super(HomographyModel, self).__init__()
#         self.hparams = hparams
#         self.model = HomographyNet(InputSize=6 * 128 * 128, OutputSize=8)
#
#     def forward(self, a, b):
#         return self.model(a, b)
#
#     def training_step(self, batch, batch_idx):
#         img_a, patch_a, patch_b, corners, gt = batch
#         delta = self.model(patch_a, patch_b)
#         loss = UnsupLossFn(delta, img_a, patch_b, corners)
#         logs = {"loss": loss}
#         return {"loss": loss, "log": logs}
#
#     def validation_step(self, batch, batch_idx):
#         img_a, patch_a, patch_b, corners, gt = batch
#         delta = self.model(patch_a, patch_b)
#         loss = LossFn(delta, img_a, patch_b, corners)
#         return {"val_loss": loss}
#
#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
#         logs = {"val_loss": avg_loss}
#         return {"avg_val_loss": avg_loss, "log": logs}
#




# Unsupervised Model

def UnsupLossFn(delta, img_a, patch_b, corners):
    """
    Compute the photometric loss for unsupervised homography estimation.
    Inputs:
        delta: Predicted H4Pt (Batch x 8)
        img_a: Image A (Batch x 128x128x3)
        patch_b: Image B (Batch x 128x128x3)
        corners: Original corners of the patch (Batch x 4 x 2)
    Outputs:
        loss: Photometric loss (scalar)
    """
    # Compute homography matrix
    H = UnsupervisedHomographyNet.get_homography(delta, corners)

    # Compute photometric loss
    loss = UnsupervisedHomographyNet.photometric_loss(img_a, patch_b, H)
    return loss
class TensorDLT(nn.Module):
    def __init__(self):
        super(TensorDLT, self).__init__()

    def forward(self, src_pts, dst_pts):

        batch_size = src_pts.size(0)
        A = torch.zeros((batch_size, 8, 9), device=src_pts.device, dtype=src_pts.dtype)

        # Reshape points
        src_pts = src_pts.view(-1, 4, 2)
        dst_pts = dst_pts.view(-1, 4, 2)

        # Create DLT matrix A
        for i in range(4):
            x, y = src_pts[:, i, 0], src_pts[:, i, 1]
            u, v = dst_pts[:, i, 0], dst_pts[:, i, 1]

            A[:, i * 2] = torch.stack([x, y, torch.ones_like(x), torch.zeros_like(x),
                                       torch.zeros_like(x), torch.zeros_like(x),
                                       -u * x, -u * y, -u], dim=1)

            A[:, i * 2 + 1] = torch.stack([torch.zeros_like(x), torch.zeros_like(x),
                                           torch.zeros_like(x), x, y, torch.ones_like(x),
                                           -v * x, -v * y, -v], dim=1)

        # Solve using SVD
        _, _, V = torch.svd(A)
        h = V[:, :, -1]
        H = h.view(-1, 3, 3)

        # Normalize homography
        H = H / H[:, 2:3, 2:3]

        return H


class UnsupervisedHomographyNet(nn.Module):
    def __init__(self, InputSize, OutputSize):
        super().__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )


        self.fc_loc = nn.Sequential(
            nn.Linear(25088, 128),  # Change 26912 → 25088
            nn.ReLU(),
            nn.Linear(128, 6)
        )


        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 8)  # Predicts 4 corner point offsets (H4Pt)
        )

        # TensorDLT layer
        self.dlt = TensorDLT()


    def forward(self, x):  # x is (batch_size, 6, 128, 128)
        """
        Forward pass of the Unsupervised Homography Model.
        Inputs:
            x: Concatenated patches (batch_size, 6, 128, 128)
        Outputs:
            H4pt: Predicted corner offsets (batch_size, 8)
        """
        x = self.stn(x)  # Apply STN
        features = self.conv_layers(x)  # CNN feature extraction
        features = features.view(features.size(0), -1)
        H4pt = self.fc_layers(features)  # Predict corner offsets
        return H4pt

    def compute_loss(self, x, H4pt, corners):
        """
        Compute the L1 loss between the predicted and ground truth corners.
        """
        predicted_corners = H4pt
        loss = F.l1_loss(predicted_corners, corners)
        return loss

    def get_homography(self, delta, corners):

        batch_size = delta.shape[0]

        # Reshape delta to (Batch x 4 x 2)
        delta = delta.view(batch_size, 4, 2)

        # Compute predicted corners
        corners_pred = corners + delta

        # Compute homography using TensorDLT
        H = self.dlt(corners, corners_pred)
        return H

    def stn(self, x):

        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        print(f"STN Output Shape before flattening: {xs.shape}")  # Debugging
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    # def stn(self, x, H):
    #     """
    #     Spatial transformer network forward function for homography estimation.
    #     Inputs:
    #         x: Input image (Batch x C x H x W)
    #         H: Homography matrix (Batch x 3 x 3)
    #     Outputs:
    #         warped: Warped image (Batch x C x H x W)
    #     """
    #     # Warp the image using OpenCV
    #     warped = warp_perspective_opencv(x, H, (x.size(2), x.size(3)))
    #     return warped


class UnsupervisedHomographyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UnsupervisedHomographyNet(InputSize=6 * 128 * 128, OutputSize=8)

    def forward(self, x):
        return self.model(x)



    def training_step(self, batch, batch_idx):

        x, corners = batch

        H4pt = self.forward(x)
        loss = self.compute_loss(x, H4pt, corners)  # Compute loss using predicted and ground truth corners

        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):
        x, corners = batch

        H4pt = self.forward(x)
        loss = self.compute_loss(x, H4pt, corners)  # Compute loss using predicted and ground truth corners
        return {"val_loss": loss}

    def compute_loss(self, x, H4pt, corners):

        predicted_corners = H4pt
        loss = F.l1_loss(predicted_corners, corners)
        return loss

