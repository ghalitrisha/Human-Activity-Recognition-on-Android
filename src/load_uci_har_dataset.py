import os
import numpy as np

def read_split_uci_dir(root_dir, split):
    base = os.path.join(root_dir, split, "Inertial Signals")
    files = [
        f"body_acc_x_{split}.txt", f"body_acc_y_{split}.txt", f"body_acc_z_{split}.txt",
        f"body_gyro_x_{split}.txt", f"body_gyro_y_{split}.txt", f"body_gyro_z_{split}.txt",
    ]
    paths = [os.path.join(base, f) for f in files]
    if not all(os.path.isfile(p) for p in paths):
        raise FileNotFoundError(f"Expected inertial signals under: {base}")

    arrays = [np.loadtxt(p) for p in paths]   # each (N,T)
    X = np.stack(arrays, axis=2).astype("float32")  # (N,T,6)

    y_path = os.path.join(root_dir, split, f"y_{split}.txt")
    if not os.path.isfile(y_path):
        raise FileNotFoundError(f"Missing labels: {y_path}")
    y = np.loadtxt(y_path).astype(int) - 1
    return X, y

def load_uci_har_dir(root_dir):
    labels_path = os.path.join(root_dir, "activity_labels.txt")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"'activity_labels.txt' not found in: {root_dir}")

    Xtr, ytr = read_split_uci_dir(root_dir, "train")
    Xte, yte = read_split_uci_dir(root_dir, "test")

    with open(labels_path, "r") as f:
        labels = [line.strip().split()[1] for line in f.readlines()]
    return Xtr, ytr, Xte, yte, labels
