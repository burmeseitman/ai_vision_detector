# Validate dataset structure for YOLO
import os, sys, glob

def check(root):
    problems = []
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        if not os.path.isdir(img_dir): problems.append(f"Missing {img_dir}")
        if not os.path.isdir(lbl_dir): problems.append(f"Missing {lbl_dir}")
    return problems

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv)>1 else "datasets/toy_car"
    issues = check(root)
    if issues:
        print("DATASET ISSUES:")
        for i in issues: print(" -", i)
        sys.exit(1)
    print("Dataset structure looks OK.")
