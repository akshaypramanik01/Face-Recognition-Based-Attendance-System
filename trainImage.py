# trainImage.py — robust getImagesAndLabels + TrainImage
import os
import cv2
import numpy as np
import csv
from glob import glob

# Allowed image extensions (lowercase)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

def _is_image_file(path):
    _, ext = os.path.splitext(path)
    return ext.lower() in _IMAGE_EXTS and os.path.isfile(path)

def getImagesAndLabels(path):
    """
    Walk `path` and collect valid image files. Return (faceSamples, Ids, label_map).
    Expected filename format: <Enrollment>_...<anything>.<ext>
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training image path not found: {path}")

    # gather files recursively to be robust (handles nested folders)
    pattern = os.path.join(path, "**", "*.*")
    files = glob(pattern, recursive=True)

    faceSamples = []
    Ids = []
    label_map = {}
    current_label = 0

    bad_files = []

    for fpath in files:
        if not _is_image_file(fpath):
            continue

        fname = os.path.basename(fpath)
        # try to derive enrollment id from filename prefix before the first underscore
        if "_" in fname:
            enrollment_id = fname.split("_", 1)[0]
        else:
            # fallback: try filename without ext
            enrollment_id = os.path.splitext(fname)[0]

        # read image
        img = cv2.imread(fpath)
        if img is None:
            bad_files.append(fpath)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Ensure we have a numeric label mapping for the enrollment
        if enrollment_id not in label_map:
            label_map[enrollment_id] = current_label
            current_label += 1

        label = label_map[enrollment_id]

        faceSamples.append(gray)
        Ids.append(label)

    # Log bad files so user can fix them
    if bad_files:
        print("[trainImage] Warning - could not read these files (skipped):")
        for bf in bad_files:
            print("   ", bf)

    if len(faceSamples) == 0:
        raise RuntimeError("No valid training images found. Check TrainingImage folder and file extensions.")

    return faceSamples, Ids, label_map


def TrainImage(haarcasecade_path, trainimage_path, trainimagelabel_path, message_label=None, text_to_speech=None):
    """
    Train LBPH model on images found under trainimage_path.
    Saves model to `trainimagelabel_path` and label_map to TrainingImageLabel/label_map.csv
    """
    try:
        faces, Ids, label_map = getImagesAndLabels(trainimage_path)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(Ids, dtype=np.int32))

        # ensure output dir exists
        out_dir = os.path.dirname(trainimagelabel_path) or "TrainingImageLabel"
        os.makedirs(out_dir, exist_ok=True)

        # save model
        recognizer.write(trainimagelabel_path)

        # save mapping (Enrollment -> Label)
        map_path = os.path.join(out_dir, "label_map.csv")
        with open(map_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Enrollment", "Label"])
            for enroll, lab in label_map.items():
                writer.writerow([enroll, lab])

        if message_label:
            try:
                message_label.config(text="Training Completed!", fg="green")
            except Exception:
                pass
        if text_to_speech:
            try:
                text_to_speech("Training Completed Successfully")
            except Exception:
                pass

        print("[trainImage] Training completed. Model saved to:", trainimagelabel_path)
        print("[trainImage] Label map saved to:", map_path)

    except Exception as exc:
        print("[trainImage] Training failed:", exc)
        if message_label:
            try:
                message_label.config(text=f"Training Failed: {exc}", fg="red")
            except Exception:
                pass
        if text_to_speech:
            try:
                text_to_speech("Training failed")
            except Exception:
                pass
        raise
