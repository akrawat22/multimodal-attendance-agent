"""Builds a small synthetic classroom video from scikit-image's sample face
(only used by the tests; needs `pip install scikit-image`)."""
import cv2
import numpy as np


def make_video(path, seconds=60, fps=10, seed=0):
    from skimage import data

    face_img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    face = face_img[40:200, 150:310]                     # head crop
    rng = np.random.default_rng(seed)
    W, H, S = 640, 360, 72
    face = cv2.resize(face, (S, S))
    seats = [(60, 40), (200, 40), (340, 40), (480, 40), (130, 190), (420, 190)]
    # (seat, appear_from, appear_to)
    plan = [(0, 0, seconds), (1, 0, seconds), (2, 0, seconds), (3, 0, seconds),
            (4, 0.65 * seconds, seconds),               # arrives late  => ~35% presence
            (5, 20, 30)]                                # walks past    => short
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in range(int(seconds * fps)):
        t = f / fps
        frame = np.full((H, W, 3), (70, 70, 70), np.uint8)
        for seat, a, b in plan:
            if not (a <= t <= b):
                continue
            if seat in (0, 1) and 20 < t < 23:         # 3 s occlusion (should be bridged)
                continue
            x, y = seats[seat]
            x += int(rng.integers(-1, 2))
            frame[y:y + S, x:x + S] = face
        vw.write(frame)
    vw.release()
    return path
