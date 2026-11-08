import cv2
import numpy as np

def merge_mertens(images):
    merger = cv2.createMergeMertens(contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.0)
    fusion = merger.process(images)
    return np.clip(fusion * 255.0, 0, 255).astype(np.uint8)

def merge_debevec(images, times, tonemap_type='reinhard'):
    times_arr = np.array(times, dtype=np.float32)
    calibrate = cv2.createCalibrateDebevec()
    crf = calibrate.process(images, times_arr)
    merge = cv2.createMergeDebevec()
    hdr_rad = merge.process(images, times_arr, crf)

    if tonemap_type == 'drago':
        tm = cv2.createTonemapDrago(gamma=1.4, saturation=1.1, bias=0.85)
    else:
        tm = cv2.createTonemapReinhard(gamma=1.5, intensity=0.0, light_adapt=0.8, color_adapt=0.2)

    ldr = tm.process(hdr_rad)
    return np.clip(ldr * 255.0, 0, 255).astype(np.uint8)
