import os
import cv2
import pandas as pd
from tqdm import tqdm

# 1. Define Paths
data_dir = 'data/raw'
csv_path = os.path.join(data_dir, 'Data_Entry_2017_v2020.csv')
raw_images_dir = 'data/1-raw/images'
output_images_dir = 'data/processed/1-normalized_images'

# Create output directory for the processed images
os.makedirs(output_images_dir, exist_ok=True)

# 2. Load and Filter the Dataset
print("Loading master CSV...")
df = pd.read_csv(csv_path)

print("Filtering for strict Pneumonia and No Finding cases...")
binary_df = df[df['Finding Labels'].isin(['Pneumonia', 'No Finding'])].copy()
print(f"Total filtered images to process: {len(binary_df)}")

# 3. Histogram Equalization Processing
print("Starting Histogram Equalization...")
processed_count = 0
missing_count = 0

# Loop through every filtered image using a progress bar
for image_name in tqdm(binary_df['Image Index']):
    input_path = os.path.join(raw_images_dir, image_name)
    output_path = os.path.join(output_images_dir, image_name)
    
    # Check if the image exists
    if not os.path.exists(input_path):
        missing_count += 1
        continue
        
    # Read the X-ray in grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        # Apply standard Histogram Equalization
        eq_img = cv2.equalizeHist(img)
        
        # Save the normalized image to the new folder
        cv2.imwrite(output_path, eq_img)
        processed_count += 1
    else:
        missing_count += 1

# 4. Save the filtered CSV so the next branches know what was processed
output_csv_path = 'data/processed/1-normalized_images/filtered.csv'
binary_df.to_csv(output_csv_path, index=False)

print("\n--- Phase 1 Complete ---")
print(f"Successfully normalized and saved: {processed_count} images")
print(f"Images missing/failed: {missing_count}")
print(f"Filtered CSV saved to: {output_csv_path}")