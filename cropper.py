#!/usr/bin/env python3

import os
import cv2
import numpy as np
import argparse

def process_folder(input_folder):
    cropped_folder = os.path.join(input_folder, "cropped")
    os.makedirs(cropped_folder, exist_ok=True)
    print(f"Saving cropped images to: {cropped_folder}") # Added feedback

    for fname in os.listdir(input_folder):
        full_path = os.path.join(input_folder, fname)
        if not os.path.isfile(full_path):
            continue
        base, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            continue

        img = cv2.imread(full_path)
        if img is None:
            print(f"  ⏭️  Could not read {fname}")
            continue

        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold to get mask: content=white, background=black
        # Adjust threshold (e.g., 230-245) if borders are not pure white
        thresh_val = 240
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # Optional: Apply morphological closing to connect slightly broken parts of content
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours of the content areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"  ⏭️  No content contours found in {fname} (threshold: {thresh_val})")
            continue

        crop_count = 0
        # Sort by area descending so largest items are processed first
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(cnt)
            # Filter tiny contours (adjust as needed)
            min_area = 5000
            if area < min_area:
                continue

            # Get the upright bounding box
            x, y, w, h = cv2.boundingRect(cnt)

            # Filter small dimension boxes (adjust as needed)
            min_dim = 50
            if w < min_dim or h < min_dim:
                continue

            # Filter boxes that are almost the entire image (likely no border)
            box_area = w * h
            if box_area / img_area > 0.95: # If box covers > 95% of image, skip
                 # print(f"    Skipping contour in {fname} (covers >95% of image area)")
                 continue # Move to next contour instead of stopping for the file

            # Crop the original image using the bounding box
            cropped_img = img[y:y+h, x:x+w]

            # Build output name and save if new
            out_name = f"{base}_crop{crop_count}{ext}"
            out_path = os.path.join(cropped_folder, out_name)

            if os.path.exists(out_path):
                print(f"    ⏩ Skipping existing {out_name}")
            else:
                cv2.imwrite(out_path, cropped_img)
                # print(f"    ✅ Saved {out_name} (Area: {area:.0f}, Box: {w}x{h})") # Verbose output
                crop_count += 1

        if crop_count > 0:
            print(f"  ✅ Processed {fname}: {crop_count} crop(s) saved")
        else:
            print(f"  ⏭️  No suitable crops found in {fname} (Min Area: {min_area}, Min Dim: {min_dim})")

def main():
    parser = argparse.ArgumentParser(
        description="Crop white borders from images." # Updated description
    )
    parser.add_argument(
        "-i", "--input-folder",
        dest="input_folder",
        required=True,
        help="Path to folder containing your images"
    )
    args = parser.parse_args()
    print(f"Processing images in: {args.input_folder}") # Added feedback
    process_folder(args.input_folder)
    print("Processing complete.") # Added feedback

if __name__ == "__main__":
    main()
