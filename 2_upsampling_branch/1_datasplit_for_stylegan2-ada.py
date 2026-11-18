"""
Stage 2: patient-level 70/20/10 split (train/test/validation) and resize 1024 -> 256 (LANCZOS).

Input:  1_global_pre-processing/output/filtered_labels.csv, normalized_images/
Output: 2_upsampling_branch/output/labels/{training,validation,testing}_labels.csv
        2_upsampling_branch/output/{training,validation,testing}/      256 x 256 images
        2_upsampling_branch/output/training_pneumonia_only/            232 images for StyleGAN2-ADA
Run:    python 2_upsampling_branch/1_datasplit_for_stylegan2-ada.py
"""
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

INPUT_IMG_DIR = os.path.join("1_global_pre-processing", "output", "normalized_images")
INPUT_CSV_PATH = os.path.join("1_global_pre-processing", "output", "filtered_labels.csv")
OUTPUT_BASE_DIR = os.path.join("2_upsampling_branch", "output")

PATIENT_ID_COL = "Patient ID"
IMAGE_NAME_COL = "Image Index"
LABEL_COL = "Finding Labels"
PNEUMONIA_LABEL = "Pneumonia"

TARGET_SIZE = (256, 256)
RANDOM_SEED = 42

def setup_directories():
    """Creates the output directory structure including the labels folder and pneumonia-only folder."""
    splits = ['training', 'validation', 'testing', 'labels', 'training_pneumonia_only']
    dirs = {}
    for split in splits:
        dir_path = os.path.join(OUTPUT_BASE_DIR, split)
        os.makedirs(dir_path, exist_ok=True)
        dirs[split] = dir_path
    return dirs

def process_and_save_images(df, split_name, output_dir, labels_dir):
    """Resizes images (grayscale) and saves them to the target directory, then saves the CSV to the labels directory."""
    print(f"Processing {split_name} set ({len(df)} images)...")

    csv_output_path = os.path.join(labels_dir, f"{split_name}_labels.csv")
    df.to_csv(csv_output_path, index=False)

    processed_count = 0
    missing_count = 0

    for _, row in df.iterrows():
        img_name = row[IMAGE_NAME_COL]
        input_path = os.path.join(INPUT_IMG_DIR, img_name)
        output_path = os.path.join(output_dir, img_name)

        if os.path.exists(input_path):
            try:
                with Image.open(input_path) as img:
                    resized_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    resized_img.save(output_path)
                    processed_count += 1
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
        else:
            missing_count += 1

    print(f"[{split_name.capitalize()}] Completed: {processed_count} resized | Missing: {missing_count}\n")

def isolate_pneumonia_images(train_df, train_dir, pneumonia_dir):
    """Copies only pneumonia images from the training set into a separate folder for StyleGAN2-ADA."""
    print("Isolating pneumonia images for StyleGAN2-ADA...")

    pneumonia_df = train_df[train_df[LABEL_COL].str.contains(PNEUMONIA_LABEL, na=False, case=False)]

    count = 0
    for _, row in pneumonia_df.iterrows():
        img_name = row[IMAGE_NAME_COL]
        src = os.path.join(train_dir, img_name)
        dst = os.path.join(pneumonia_dir, img_name)

        if os.path.exists(src):
            shutil.copy2(src, dst)
            count += 1

    print(f"Isolated {count} pneumonia images into: {pneumonia_dir}\n")

def main():
    output_dirs = setup_directories()

    df = pd.read_csv(INPUT_CSV_PATH)

    # Patient-level split (70% train, then the remaining 30% split 2:1 into test and validation)
    unique_patients = df[PATIENT_ID_COL].unique()

    train_patients, temp_patients = train_test_split(
        unique_patients, test_size=0.30, random_state=RANDOM_SEED
    )
    test_patients, val_patients = train_test_split(
        temp_patients, test_size=(1/3), random_state=RANDOM_SEED
    )

    train_df = df[df[PATIENT_ID_COL].isin(train_patients)]
    test_df  = df[df[PATIENT_ID_COL].isin(test_patients)]
    val_df   = df[df[PATIENT_ID_COL].isin(val_patients)]

    print(f"Total Unique Patients: {len(unique_patients)}")
    print(f"Train Patients: {len(train_patients)} | Images: {len(train_df)}")
    print(f"Test Patients:  {len(test_patients)} | Images: {len(test_df)}")
    print(f"Val Patients:   {len(val_patients)} | Images: {len(val_df)}\n")

    labels_dir = output_dirs['labels']
    process_and_save_images(train_df, 'training', output_dirs['training'], labels_dir)
    process_and_save_images(test_df,  'testing',  output_dirs['testing'],  labels_dir)
    process_and_save_images(val_df,   'validation', output_dirs['validation'], labels_dir)

    # Pneumonia-only training images: the StyleGAN2-ADA training set
    isolate_pneumonia_images(train_df, output_dirs['training'], output_dirs['training_pneumonia_only'])

    print(f"All done. Output written to: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()