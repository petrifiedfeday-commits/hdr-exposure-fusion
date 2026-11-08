import cv2
import numpy as np
from hdr.merging import merge_mertens

def detect_motion_mask(images, ref_idx=1, threshold=25):
    ref_norm = cv2.equalizeHist(cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2GRAY))
    h, w = ref_norm.shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for i, img in enumerate(images):
        if i == ref_idx:
            continue
        tgt_norm = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        diff = cv2.absdiff(ref_norm, tgt_norm)
        _, b_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_OPEN, kernel)
        b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.bitwise_or(combined_mask, b_mask)

    return cv2.GaussianBlur(combined_mask, (21, 21), 0)

def merge_mertens_deghosted(images, ref_idx=1, threshold=25):
    hdr_raw = merge_mertens(images)
    motion_mask = detect_motion_mask(images, ref_idx=ref_idx, threshold=threshold)
    alpha = np.dstack([motion_mask.astype(np.float32) / 255.0] * 3)

    ref_f = images[ref_idx].astype(np.float32)
    hdr_f = hdr_raw.astype(np.float32)

    deghosted = alpha * ref_f + (1.0 - alpha) * hdr_f
    return np.clip(deghosted, 0, 255).astype(np.uint8), motion_mask
