import os
 
DIRS = [
    "models/densenet121_cbam",
    "results/classification/densenet121_cbam",
    "results/classification/stats_tests",
    "scripts/classification",
]
 
if __name__ == "__main__":
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
        print(f"Created (or already exists): {d}")