import time
import cv2
import numpy as np
from collections import deque
from hdr.alignment import align_mtb
from hdr.merging import merge_mertens

class RealTimeHDRProcessor:
    def __init__(self, buffer_size=3, scale=0.75):
        self.buffer = deque(maxlen=buffer_size)
        self.scale = scale
        self.merger = cv2.createMergeMertens()
        self.fps_records = []

    def process(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w*self.scale), int(h*self.scale)))
        self.buffer.append(small)

        if len(self.buffer) < self.buffer.maxlen:
            return small, 0.0

        t0 = time.perf_counter()
        aligned = align_mtb(list(self.buffer))
        hdr = self.merger.process(aligned)
        hdr_u8 = np.clip(hdr * 255.0, 0, 255).astype(np.uint8)
        dt = time.perf_counter() - t0

        fps = 1.0 / dt if dt > 0 else 0.0
        self.fps_records.append(fps)
        return hdr_u8, fps
