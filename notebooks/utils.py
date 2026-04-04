"""
Utility functions cho notebook 02_preprocessing_image.ipynb
Tập trung các hàm load dữ liệu dùng chung để tránh lặp code.
"""

import os
import glob
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# 1. Thiết lập đường dẫn & lấy danh sách ảnh
# ============================================================

def get_image_paths(directory):
    """Lấy toàn bộ đường dẫn ảnh (.jpg, .png) trong *directory*."""
    return (
        glob.glob(os.path.join(directory, "*", "*.jpg"))
        + glob.glob(os.path.join(directory, "*", "*.png"))
    )


def load_all_image_paths(data_root="."):
    """
    Trả về (train_paths, val_paths, test_paths) từ thư mục *data_root*.
    Mỗi phần tử là list đường dẫn tuyệt đối tới các ảnh.
    """
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")
    test_dir  = os.path.join(data_root, "test")

    train_paths = get_image_paths(train_dir)
    val_paths   = get_image_paths(val_dir)
    test_paths  = get_image_paths(test_dir)

    return train_paths, val_paths, test_paths


# ============================================================
# 2. Load & tiền xử lý ảnh
# ============================================================

def load_imgs(paths, size=(64, 64), cs="RGB"):
    """
    Load, resize, convert color space, flatten images.
    
    Parameters
    ----------
    paths : list[str]
        Danh sách đường dẫn ảnh.
    size : tuple[int, int]
        Kích thước resize (width, height).
    cs : str
        Không gian màu: 'RGB', 'Grayscale', 'HSV', 'LAB'.
    
    Returns
    -------
    (X, y) : tuple[np.ndarray, np.ndarray]
        X – mảng 2‑D (N, D) chứa ảnh đã flatten, dtype=float32.
        y – mảng 1‑D (N,) chứa nhãn (tên thư mục cha).
    """
    imgs, labels = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, size)
        if cs == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif cs == "Grayscale":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif cs == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif cs == "LAB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        imgs.append(img.flatten().astype(np.float32))
        labels.append(os.path.basename(os.path.dirname(p)))
    return np.array(imgs), np.array(labels)


def load_imgs_raw(paths, size=(128, 128), cs="RGB"):
    """
    Load ảnh giữ nguyên shape (N, H, W, C) – không flatten.
    
    Parameters
    ----------
    paths : list[str]
        Danh sách đường dẫn ảnh.
    size : tuple[int, int]
        Kích thước resize (width, height).
    cs : str
        Không gian màu: 'RGB' (mặc định).
    
    Returns
    -------
    np.ndarray, dtype=float32, shape=(N, H, W, C)
    """
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            if cs == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            images.append(img)
    return np.array(images, dtype=np.float32)


# ============================================================
# 3. Đánh giá bằng Logistic Regression + PCA
# ============================================================

def eval_lr_pca(X, y, n_comp=50, cv=3, seed=42):
    """
    PCA + StandardScaler + LogisticRegression (saga, max_iter=3000).
    Returns (mean_accuracy, std_accuracy) via Stratified K-Fold CV.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    nc = min(n_comp, X.shape[0] - 1, X.shape[1])
    pca_ = PCA(n_components=nc, random_state=seed)
    X_pca = pca_.fit_transform(X)
    scaler_ = StandardScaler()
    X_scaled = scaler_.fit_transform(X_pca)
    lr_ = LogisticRegression(max_iter=3000, random_state=seed, solver="saga")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = cross_val_score(lr_, X_scaled, y_enc, cv=skf, scoring="accuracy")
    return scores.mean(), scores.std()
