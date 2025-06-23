import os
from copy import deepcopy

import sys
import torch
import cv2
import numpy as np
import time
from glob import glob
from src.loftr import LoFTR, opt_default_cfg
import re

def concatenate_filenames(paths):
    filenames = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    concatenated_filenames = '_'.join(filenames)
    return concatenated_filenames

def increment_image_path(path):
    dirname, filename = os.path.split(path)

    match = re.search(r"(\d+)(\.\w+)$", filename)
    if not match:
        return None

    number_str, extension = match.groups()
    number = int(number_str)
    new_number_str = f"{number + 1:06d}"  # Keep same padding

    new_filename = new_number_str + extension
    return os.path.join(dirname, new_filename)

# Check if the correct number of arguments are provided
if len(sys.argv) < 5:
    print("Usage: python loftr_batch_matcher.py weights_file img_query img0 ... imgN path_to_output")
    sys.exit(1)

TARGET_SIZE = (1536, 768)
# --- Initialize model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matcher = LoFTR(config=opt_default_cfg)
matcher.load_state_dict(torch.load(sys.argv[1], map_location=device)['state_dict'])
matcher = matcher.to(device).eval()

img_query_path = sys.argv[2]
img_list = sys.argv[3:-1]
output_path = sys.argv[-1]
print(img_query_path)
print(img_list)
print(sys.argv[-1])

# match consecutive images first
next_image = increment_image_path(img_query_path)
print(f"Next image to match: {next_image}")

img_query = cv2.imread(img_query_path, cv2.IMREAD_GRAYSCALE)
if img_query is None:
    print(f"Failed to load query image: {img_query_path}")
    exit(1) # exit if query image is not found

img_query_next = cv2.imread(next_image, cv2.IMREAD_GRAYSCALE)
if not (img_query_next is None): 
    # create consecutive matches
    print(f"Matching: {img_query_path} ↔ {next_image}")
    img_query = cv2.resize(img_query, TARGET_SIZE)
    img_query_next = cv2.resize(img_query_next, TARGET_SIZE)
    # Prepare input tensors
    img_query_tensor = torch.from_numpy(img_query)[None][None].float() / 255.0
    img_query_next_tensor = torch.from_numpy(img_query_next)[None][None].float() / 255.0
    with torch.no_grad():
        input_dict = {"image0": img_query_tensor.to(device), "image1": img_query_next_tensor.to(device)}
        matcher(input_dict)
        mkpts0 = input_dict['mkpts0_f'].cpu().numpy()
        mkpts1 = input_dict['mkpts1_f'].cpu().numpy()
    print(f"Matches found: {len(mkpts0)}")

    tmpfile = output_path + concatenate_filenames([img_query_path, next_image]) + "_x" + ".tmp"
    finalfile = output_path + concatenate_filenames([img_query_path, next_image]) + "_x" + ".txt"
    with open(tmpfile, 'w') as f:
        f.write(str(mkpts0.shape[0]) + "\n")
        scaled_data0 = np.copy(mkpts0)
        scaled_data0[:, 0] /= img_query.shape[1]
        scaled_data0[:, 1] /= img_query.shape[0]
        np.savetxt(f, scaled_data0, fmt='%.6f', delimiter=' ')
        f.write(str(mkpts1.shape[0]) + "\n")
        scaled_data1 = np.copy(mkpts1)
        scaled_data1[:, 0] /= img_query_next.shape[1]
        scaled_data1[:, 1] /= img_query_next.shape[0]
        np.savetxt(f, scaled_data1, fmt='%.6f', delimiter=' ')
    os.rename(tmpfile, finalfile)

# match all other images
cnt = 0
for img in img_list:
    img1_pth = img
    img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)

    if img1_raw is None:
        print(f"Failed to load ref image: {img1_pth}")
        exit(1)

    print(f"Matching: {img_query_path} ↔ {img1_pth}")
    img_query = cv2.resize(img_query, TARGET_SIZE)
    img_ref = cv2.resize(img1_raw, TARGET_SIZE)
    # Prepare input tensors
    img_query_tensor = torch.from_numpy(img_query)[None][None].float() / 255.0
    img_ref_tensor = torch.from_numpy(img_ref)[None][None].float() / 255.0

    with torch.no_grad():
        input_dict = {"image0": img_query_tensor.to(device), "image1": img_ref_tensor.to(device)}
        matcher(input_dict)
        mkpts0 = input_dict['mkpts0_f'].cpu().numpy()
        mkpts1 = input_dict['mkpts1_f'].cpu().numpy()
    print(f"Matches found: {len(mkpts0)}")

    tmpfile = output_path + concatenate_filenames([img_query_path, img1_pth]) + "_" + str(cnt) + ".tmp"
    finalfile = output_path + concatenate_filenames([img_query_path, img1_pth]) + "_" + str(cnt) + ".txt"

    with open(tmpfile, 'w') as f:
        f.write(str(mkpts0.shape[0]) + "\n")
        scaled_data0 = np.copy(mkpts0)
        scaled_data0[:, 0] /= img_query.shape[1]
        scaled_data0[:, 1] /= img_query.shape[0]
        np.savetxt(f, scaled_data0, fmt='%.6f', delimiter=' ')
        f.write(str(mkpts1.shape[0]) + "\n")
        scaled_data1 = np.copy(mkpts1)
        scaled_data1[:, 0] /= img_ref.shape[1]
        scaled_data1[:, 1] /= img_ref.shape[0]
        np.savetxt(f, scaled_data1, fmt='%.6f', delimiter=' ')

    os.rename(tmpfile, finalfile)
    cnt += 1