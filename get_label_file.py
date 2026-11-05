import csv
import os
import random

# ---- CONFIGURE HERE ----
image_root = "/home/guests/jsm/isbi_paper/kvasir_seg_dataset/images_512"  # image folder
mask_root = (
    "/home/guests/jsm/isbi_paper/kvasir_seg_dataset/masks_binary_512"  # mask folder
)
OUTPUT_CSV = "./kvasir_pairs_30folds.csv"  # new name: do not overwrite the CSVs in use
TOTAL_FOLDS = 30
FOLD_SEED = 0  # same seed → same folds every time
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
# -------------------------

rows = []
for image_name in sorted(os.listdir(image_root)):
    image_path = os.path.join(image_root, image_name)
    if not os.path.isfile(image_path) or not image_name.lower().endswith(
        IMAGE_EXTENSIONS
    ):
        continue

    mask_path = os.path.join(mask_root, image_name)
    if not os.path.exists(mask_path):
        print(f"Skipping {image_name}: mask file not found.")
        continue

    rows.append([image_name, mask_path, image_path])

if len(rows) < TOTAL_FOLDS:
    raise ValueError(f"Only {len(rows)} images for {TOTAL_FOLDS} folds")

# Random but reproducible folds of equal size (differ by at most 1 image)
random.Random(FOLD_SEED).shuffle(rows)
for idx, row in enumerate(rows):
    row.append(idx % TOTAL_FOLDS)

# Write CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["wsi_name", "mask_path", "tile_path", "fold"])
    writer.writerows(rows)

print(f"Saved {len(rows)} rows with {TOTAL_FOLDS} folds to {OUTPUT_CSV}")
