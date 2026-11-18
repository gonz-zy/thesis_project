"""
Stage 1: keeps only Pneumonia and No Finding images and applies histogram equalization.

Input:  0_dataset/Data_Entry_2017_v2020.csv, 0_dataset/images/
Output: 1_global_pre-processing/output/filtered_labels.csv
        1_global_pre-processing/output/normalized_images/   (1024 x 1024, equalized)
Run:    python 1_global_pre-processing/1_histogram_normalization.py
"""
import os
import cv2
import pandas as pd
from tqdm import tqdm

csv_path = os.path.join('0_dataset', 'Data_Entry_2017_v2020.csv')
raw_images_dir = os.path.join('0_dataset', 'images')
output_images_dir = os.path.join('1_global_pre-processing', 'output', 'normalized_images')
output_csv_path = os.path.join('1_global_pre-processing', 'output', 'filtered_labels.csv')

os.makedirs(output_images_dir, exist_ok=True)

# Label filtering: keep only Pneumonia and No Finding
print("Loading master CSV...")
df = pd.read_csv(csv_path)

print("Filtering for strict Pneumonia and No Finding cases...")
binary_df = df[df['Finding Labels'].isin(['Pneumonia', 'No Finding'])].copy()
print(f"Total filtered images to process: {len(binary_df)}")

# Histogram equalization
print("Starting Histogram Equalization...")
processed_count = 0
missing_count = 0

for image_name in tqdm(binary_df['Image Index']):
    input_path = os.path.join(raw_images_dir, image_name)
    output_path = os.path.join(output_images_dir, image_name)
    
    if not os.path.exists(input_path):
        missing_count += 1
        continue
        
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        eq_img = cv2.equalizeHist(img)
        
        cv2.imwrite(output_path, eq_img)
        processed_count += 1
    else:
        missing_count += 1

binary_df.to_csv(output_csv_path, index=False)

print("\n--- Phase 1 Complete ---")
print(f"Successfully normalized and saved: {processed_count} images")
print(f"Images missing/failed: {missing_count}")
print(f"Filtered CSV saved to: {output_csv_path}")