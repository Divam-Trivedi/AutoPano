#!/usr/bin/env python

import numpy as np
import cv2
import torch
import argparse
import os
from Network.Network import UnsupervisedHomographyModel
import matplotlib.pyplot as plt
import json


def visualize_crops(img, original_corners, perturbed_corners, predicted_corners, save_path=None):
    """Visualize different crops with color-coded boxes"""
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    corners_list = [
        (original_corners, 'red', 'Original Crop'),
        (perturbed_corners, 'blue', 'Perturbed Crop'),
        (predicted_corners, 'yellow', 'Unsupervised Crop')
    ]

    for corners, color, label in corners_list:
        for i in range(4):
            j = (i + 1) % 4
            plt.plot([corners[i][0], corners[j][0]],
                     [corners[i][1], corners[j][1]],
                     color=color, linewidth=2, label=label if i == 0 else None)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def process_image_pair(model, img1, img2):
    """Process a single image pair through the model"""
    device = next(model.parameters()).device
    h1, w1 = img1.shape[:2]

    # Resize and stack images
    img1_resized = cv2.resize(img1, (128, 128))
    img2_resized = cv2.resize(img2, (128, 128))
    stacked = np.concatenate([img1_resized, img2_resized], axis=2)

    # Prepare tensor
    img_tensor = torch.from_numpy(stacked.astype(np.float32) / 255.0)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        h4pt = model(img_tensor)
        H = model.model.dlt(h4pt, torch.ones_like(h4pt)).cpu().numpy()[0]

    # Scale homography to original image size
    scale_mat = np.array([[w1 / 128, 0, 0],
                          [0, h1 / 128, 0],
                          [0, 0, 1]])
    H = scale_mat @ H @ np.linalg.inv(scale_mat)

    return H


def blend_images(img1, img2, H):
    """Blend two images using computed homography"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Find corners of warped image
    pts = np.float32([[0, 0], [w2 - 1, 0], [w2 - 1, h2 - 1], [0, h2 - 1]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

    [xmin, ymin] = np.int32(dst.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(dst.max(axis=0).ravel() + 0.5)

    # Translation matrix
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    # Warp images
    size = (xmax - xmin, ymax - ymin)
    warped1 = cv2.warpPerspective(img1, Ht, size)
    warped2 = cv2.warpPerspective(img2, Ht @ H, size)

    # Create masks
    mask1 = (cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY) > 0)
    mask2 = (cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY) > 0)

    # Blend images
    result = warped1.copy()
    overlap = mask1 & mask2
    if overlap.any():
        for c in range(3):
            result[overlap, c] = (warped1[overlap, c] * 0.5 +
                                  warped2[overlap, c] * 0.5)
        result[~mask1 & mask2] = warped2[~mask1 & mask2]

    return result


def evaluate_homography(model, test_json_path, output_dir):
    """Evaluate homography estimation"""
    device = next(model.parameters()).device
    model.eval()

    with open(test_json_path, 'r') as f:
        test_data = json.load(f)

    total_error = 0
    results = {}

    for idx, data in test_data.items():
        # Create output directory
        name = data['img_name']
        set_dir = os.path.join(output_dir, 'eval', os.path.splitext(name)[0])
        os.makedirs(set_dir, exist_ok=True)

        # Get input data
        stacked = np.array(data['stacked_input1'])
        gt_h4pt = np.array(data['H4pt_input2'])
        original_corners = np.array(data['cornersA']).reshape(-1, 2)
        perturbed_corners = np.array(data['cornersB']).reshape(-1, 2)

        # Ensure input is float32 and normalized
        stacked = stacked.astype(np.float32)
        if stacked.max() > 1.0:
            stacked = stacked / 255.0

        # Resize if needed
        if stacked.shape[:2] != (128, 128):
            resized = np.zeros((128, 128, 6), dtype=np.float32)
            resized[:, :, :3] = cv2.resize(stacked[:, :, :3], (128, 128))
            resized[:, :, 3:] = cv2.resize(stacked[:, :, 3:], (128, 128))
            stacked = resized

        # Convert to tensor
        img_tensor = torch.from_numpy(stacked).permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                pred_h4pt = model(img_tensor)
                pred_h4pt = pred_h4pt.cpu().numpy().reshape(-1, 2)

                # Scale predicted points back to original image size if needed
                if stacked.shape[:2] != (128, 128):
                    h, w = stacked.shape[:2]
                    pred_h4pt[:, 0] *= w / 128
                    pred_h4pt[:, 1] *= h / 128

            # Calculate error
            error = np.mean(np.sqrt(np.sum((pred_h4pt - gt_h4pt.reshape(-1, 2)) ** 2, axis=1)))
            total_error += error

            # Visualize results
            img_a = stacked[:, :, :3]
            if img_a.max() <= 1.0:
                img_a = (img_a * 255).astype(np.uint8)

            save_path = os.path.join(set_dir, f'{name}_vis.png')
            visualize_crops(img_a, original_corners, perturbed_corners, pred_h4pt, save_path)

            results[name] = {
                'error': float(error),
                'predicted_corners': pred_h4pt.tolist()
            }

        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
            continue

    avg_error = total_error / len(results)
    print(f'Average EPE: {avg_error:.4f}')

    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump({
            'average_error': float(avg_error),
            'results': results
        }, f, indent=2)

    return avg_error


def get_sequential_pairs(image_path):

    pairs = {}
    for filename in sorted(os.listdir(image_path)):
        if not filename.endswith('.jpg'):
            continue

        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')

        if len(parts) < 2:
            continue

        main_seq = parts[0]
        sub_seq = '.'.join(parts[:2])

        if main_seq not in pairs:
            pairs[main_seq] = {}
        if sub_seq not in pairs[main_seq]:
            pairs[main_seq][sub_seq] = []

        pairs[main_seq][sub_seq].append(filename)

    return pairs


def stitch_panorama(model, image_path, output_dir):
    pano_dir = os.path.join(output_dir, 'panoramas')
    os.makedirs(pano_dir, exist_ok=True)

    # Group images by main sequence number
    sequences = {}
    for filename in sorted(os.listdir(image_path)):
        if not filename.endswith('.jpg'):
            continue
        seq_num = filename.split('.')[0]
        if seq_num not in sequences:
            sequences[seq_num] = []
        sequences[seq_num].append(filename)

    for seq_num, files in sequences.items():
        files.sort()  # Ensure proper ordering
        print(f"\nProcessing sequence {seq_num}")
        seq_dir = os.path.join(pano_dir, f'sequence_{seq_num}')
        os.makedirs(seq_dir, exist_ok=True)

        if len(files) < 2:
            continue

        # Initialize with first image
        base_img = cv2.imread(os.path.join(image_path, files[0]))
        cv2.imwrite(os.path.join(seq_dir, 'step_0.jpg'), base_img)

        # Sequentially stitch remaining patches
        for i in range(1, len(files)):
            curr_img = cv2.imread(os.path.join(image_path, files[i]))
            if curr_img is None:
                continue

            try:
                H = process_image_pair(model, base_img, curr_img)
                base_img = blend_images(base_img, curr_img, H)
                cv2.imwrite(os.path.join(seq_dir, f'step_{i}.jpg'), base_img)
            except Exception as e:
                print(f"Error processing {files[i]}: {str(e)}")
                continue

        cv2.imwrite(os.path.join(seq_dir, 'panorama.jpg'), base_img)

        print(f"Completed panorama for sequence")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ModelPath', default='D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p1\\YourDirectoryID_p1\\Phase2\\Checkpoints\\Val\\49model.ckpt')
    parser.add_argument('--TestJsonPath', default='D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p1\\YourDirectoryID_p1\\Phase2\\Data\\val__json.json')
    parser.add_argument('--InputImagesPath', default='D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p1\\YourDirectoryID_p1\\Phase2\\Data\\Val_patch\\Val')
    parser.add_argument('--OutputDir', default='Results')

    parser.add_argument('--ModelType', default='Unsup')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UnsupervisedHomographyModel()
    model = model.to(device)

    checkpoint = torch.load(args.ModelPath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Evaluating homography estimation...")
    avg_error = evaluate_homography(model, args.TestJsonPath, args.OutputDir)

    print("\nStitching panorama...")
    stitch_panorama(model, args.InputImagesPath, args.OutputDir)

    print("Processing complete!")

if __name__ == "__main__":
    main()