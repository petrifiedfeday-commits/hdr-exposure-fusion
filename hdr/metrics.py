import cv2
import numpy as np

def calculate_clipping_percentages(img, low_thresh=5, high_thresh=250):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    total = gray.size
    u_pct = (np.sum(gray <= low_thresh) / total) * 100.0
    o_pct = (np.sum(gray >= high_thresh) / total) * 100.0
    return float(u_pct), float(o_pct)

def calculate_shannon_entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    p = hist[hist > 0] / hist.sum()
    return float(-np.sum(p * np.log2(p)))

def get_image_metrics(img):
    u_pct, o_pct = calculate_clipping_percentages(img)
    ent = calculate_shannon_entropy(img)
    return {
        "underexposed_pct": u_pct,
        "overexposed_pct": o_pct,
        "clipping_total_pct": u_pct + o_pct,
        "entropy": ent
    }
