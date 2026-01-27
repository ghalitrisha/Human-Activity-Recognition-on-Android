import os
import numpy as np
from collections import Counter
from typing import List, Tuple, Optional


def find_wisdm_raw_txt(root_dir: str) -> Optional[str]:
    for f in os.listdir(root_dir):
        name = f.lower()
        if name.endswith(".txt") and "wisdm_ar_v1.1_raw" in name:
            return os.path.join(root_dir, f)
    for f in os.listdir(root_dir):
        name = f.lower()
        if name.endswith(".txt") and "wisdm" in name and "raw" in name:
            return os.path.join(root_dir, f)
    return None

def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def parse_wisdm_raw(root_dir: str):
    """
    Returns per-sample arrays:
      users: (N,) int
      labels: (N,) str (lowercased activity)
      ax, ay, az: (N,) float32
    """
    txt = find_wisdm_raw_txt(root_dir)
    if txt is None:
        raise FileNotFoundError(f"No WISDM v1.1 raw .txt found in: {root_dir}")

    users, labels, ax, ay, az = [], [], [], [], []

    with open(txt, "r", errors="ignore") as f:
        for line in f:
            # lines can end with ';' and have variable spacing
            line = line.strip().replace(";", ",")
            if not line:
                continue
            parts = [p for p in line.split(",") if p.strip() != ""]
            # Expected (common) layout: user, activity, timestamp, ax, ay, az
            if len(parts) < 6:
                continue

            # user id (int-ish)
            uid_raw = parts[0].strip()
            try:
                uid = int(uid_raw)
            except Exception:
                # Sometimes first col can have prefix; try to strip non-digits
                digits = "".join(ch for ch in uid_raw if ch.isdigit())
                if digits == "":
                    continue
                uid = int(digits)

            label = parts[1].strip().lower()

            # accel values usually last 3 fields
            x = safe_float(parts[-3]); y = safe_float(parts[-2]); z = safe_float(parts[-1])
            if x is None or y is None or z is None:
                continue

            users.append(uid); labels.append(label)
            ax.append(x); ay.append(y); az.append(z)

    if len(ax) == 0:
        raise ValueError(f"No valid rows parsed from {txt}")

    users  = np.asarray(users,  dtype=np.int32)
    labels = np.asarray(labels, dtype=object)
    ax     = np.asarray(ax,     dtype=np.float32)
    ay     = np.asarray(ay,     dtype=np.float32)
    az     = np.asarray(az,     dtype=np.float32)
    return users, labels, ax, ay, az


def make_windows(
    users: np.ndarray,
    labels: np.ndarray,
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    window: int = 128,
    step: int = 64
):
    """
    Builds windows of shape (Nw, window, 6). Last 3 channels are zeros
    to match UCI HAR's (acc+gyro) layout.
    Returns: X, y, win_users
    """
    N = len(ax)
    X_list, y_list, u_list = [], [], []

    zeros = np.zeros((window, 3), dtype=np.float32)

    for start in range(0, N - window + 1, step):
        end = start + window

        # Majority label and user within the window
        maj_label = Counter(labels[start:end]).most_common(1)[0][0]
        maj_user  = Counter(users[start:end]).most_common(1)[0][0]

        seg = np.stack([ax[start:end], ay[start:end], az[start:end]], axis=1)  # (T,3)
        X6  = np.concatenate([seg, zeros], axis=1).astype(np.float32)         # (T,6)

        X_list.append(X6)
        y_list.append(maj_label)
        u_list.append(maj_user)

    if not X_list:
        X = np.zeros((0, window, 6), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)
        win_users = np.zeros((0,), dtype=np.int32)
        labels_map = []
        return X, y, win_users, labels_map

    # map labels -> ids (stable lexicographic order for reproducibility)
    uniq = sorted(set(y_list))
    lbl2id = {lbl:i for i,lbl in enumerate(uniq)}

    X = np.stack(X_list, axis=0)                       # (Nw, T, 6)
    y = np.array([lbl2id[lbl] for lbl in y_list], np.int64)
    win_users = np.asarray(u_list, dtype=np.int32)
    labels_map = [None]*len(lbl2id)
    for k,v in lbl2id.items():
        labels_map[v] = k
    return X, y, win_users, labels_map

# ---------- public API (like UCI) ----------

def load_wisdm_dir(
    root_dir: str,
    window: int = 128,
    step: int = 64,
    split_mode: str = "user",           # "user" or "random"
    test_user_ids: Optional[List[int]] = None,
    test_ratio: float = 0.2,            # only used if split_mode="random"
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Returns: X_train, y_train, X_test, y_test, labels  (same contract as UCI loader)
    Shapes:  (N, T, 6) with T=window; labels = list[str] id->name
    """
    users, labels_raw, ax, ay, az = parse_wisdm_raw(root_dir)
    X_all, y_all, win_users, labels_map = make_windows(users, labels_raw, ax, ay, az, window, step)

    if split_mode not in ("user", "random"):
        raise ValueError("split_mode must be 'user' or 'random'")

    if split_mode == "user":
        # choose test users
        if not test_user_ids:
            # default: use the last ~20% of users (sorted) as test
            uniq_users = sorted(np.unique(win_users).tolist())
            k = max(1, int(0.2*len(uniq_users)))
            test_user_ids = uniq_users[-k:]
        test_user_ids = set(int(u) for u in test_user_ids)

        test_mask  = np.isin(win_users, list(test_user_ids))
        train_mask = ~test_mask

    else:
        # stratified random split by label
        rng = np.random.RandomState(seed)
        train_mask = np.zeros(len(y_all), dtype=bool)
        test_mask  = np.zeros(len(y_all), dtype=bool)
        for c in np.unique(y_all):
            idx = np.where(y_all == c)[0]
            rng.shuffle(idx)
            n_test = int(round(test_ratio * len(idx)))
            test_idx = idx[:n_test]
            train_idx = idx[n_test:]
            test_mask[test_idx] = True
            train_mask[train_idx] = True

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    return X_train, y_train, X_test, y_test, labels_map
