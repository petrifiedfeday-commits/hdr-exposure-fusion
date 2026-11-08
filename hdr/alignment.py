import cv2
import numpy as np

def align_mtb(images, ref_idx=1):
    aligner = cv2.createAlignMTB(max_bits=6, exclude_range=4)
    ref_g = cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2GRAY)
    aligned = []
    for i, img in enumerate(images):
        if i == ref_idx:
            aligned.append(img.copy())
        else:
            tgt_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            shift = aligner.calculateShift(ref_g, tgt_g)
            aligned.append(aligner.shiftMat(img, shift))
    return aligned

def align_keypoints_homography(images, ref_idx=1):
    detector = cv2.SIFT_create(nfeatures=2000)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    def prep(im):
        return clahe.apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

    ref_img = images[ref_idx]
    h, w = ref_img.shape[:2]
    kp_ref, des_ref = detector.detectAndCompute(prep(ref_img), None)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    aligned = [None] * len(images)
    aligned[ref_idx] = ref_img.copy()

    for i in range(len(images)):
        if i == ref_idx:
            continue
        tgt = images[i]
        kp_tgt, des_tgt = detector.detectAndCompute(prep(tgt), None)
        raw = matcher.knnMatch(des_tgt, des_ref, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]

        pts_tgt = np.float32([kp_tgt[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(pts_tgt, pts_ref, cv2.RANSAC, 3.0)
        aligned[i] = cv2.warpPerspective(tgt, H, (w, h), borderMode=cv2.BORDER_REFLECT)

    return aligned
