import os
import cv2
import sys
import torch
import numpy as np
from glob import glob
from src.loftr import LoFTR, opt_default_cfg

TARGET_SIZE = (1536, 768)

# --- Initialize model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matcher = LoFTR(config=opt_default_cfg)
matcher.load_state_dict(torch.load(sys.argv[4], map_location=device)['state_dict'])
matcher = matcher.to(device).eval()

def concatenate_filenames(paths):
    filenames = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    concatenated_filenames = '_'.join(filenames)
    return concatenated_filenames

def parse_and_match(txt_path, input_dir1, input_dir2):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        header = lines[i].strip().split()
        query_id = int(header[0])
        num_refs = int(header[1])
        i += 1

        # Load query image
        query_path = os.path.join(input_dir1, f"{query_id:06d}.jpg")
        img_query = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
        if img_query is None:
            print(f"Failed to load query image: {query_path}")
            i += num_refs
            continue

        # load next query image
        query_path_next = os.path.join(input_dir1, f"{query_id + 1:06d}.jpg")
        img_query_next = cv2.imread(query_path_next, cv2.IMREAD_GRAYSCALE)
        if not (img_query_next is None): 
            # create consecutive matches
            print(f"Matching: {query_path} ↔ {query_path_next}")
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
            outfile = sys.argv[2] + concatenate_filenames([query_path, query_path_next]) + "_x.txt"
            with open(outfile, 'w') as f:
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

        # Process reference entries
        for j in range(num_refs):
            ref_line = lines[i + j].strip().split()
            ref_subdir = ref_line[0].split('/')[0] + "/images"
            ref_id = int(ref_line[1])
            score = float(ref_line[2])  # You can use this if needed

            ref_path = os.path.join(input_dir2, ref_subdir, f"{ref_id:06d}.jpg")
            img_ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if img_ref is None:
                print(f"Failed to load ref image: {ref_path}")
                continue

            print(f"Matching: {query_path} ↔ {ref_path}")

            img_query = cv2.resize(img_query, TARGET_SIZE)
            img_ref = cv2.resize(img_ref, TARGET_SIZE)
            # Prepare input tensors
            img_query_tensor = torch.from_numpy(img_query)[None][None].float() / 255.0
            img_ref_tensor = torch.from_numpy(img_ref)[None][None].float() / 255.0

            with torch.no_grad():
                input_dict = {"image0": img_query_tensor.to(device), "image1": img_ref_tensor.to(device)}
                matcher(input_dict)
                mkpts0 = input_dict['mkpts0_f'].cpu().numpy()
                mkpts1 = input_dict['mkpts1_f'].cpu().numpy()

            print(f"Matches found: {len(mkpts0)}")

            outfile = sys.argv[2] + concatenate_filenames([query_path, ref_path]) + "_" + str(j) + ".txt"

            with open(outfile, 'w') as f:
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

            if False:
                drawn = cv2.drawMatches(
                    cv2.cvtColor(img_query, cv2.COLOR_GRAY2BGR), [cv2.KeyPoint(p[0], p[1], 1) for p in mkpts0],
                    cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR), [cv2.KeyPoint(p[0], p[1], 1) for p in mkpts1],
                    [cv2.DMatch(j, j, 0) for j in range(len(mkpts0))],
                    None
                )

                out_path = f"match_{i-1:04d}_{i:04d}.jpg"
                cv2.imwrite(out_path, drawn)

        # Move to the next query block
        i += num_refs

    print("Done parsing and matching.")

# Usage
parse_and_match(
    txt_path=sys.argv[1],
    input_dir1=sys.argv[2],
    input_dir2=sys.argv[3]
)
