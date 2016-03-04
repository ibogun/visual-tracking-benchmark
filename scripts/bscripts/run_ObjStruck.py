import subprocess
import time
import numpy as np

from trackers.Antrack.antrack import ObjStruck
from config import *


def run_ObjStruck(seq, rp, bSaveImage):
    x = seq.init_rect[0] - 1
    y = seq.init_rect[1] - 1
    w = seq.init_rect[2]
    h = seq.init_rect[3]

    path = './results/'

    if not os.path.exists(path):
        os.makedirs(path)

    # x, y, w, h -> initial bounding box
    # seq.s_frames is a list of the images
    features = "hogANDhist"
    kernel = "int"
    filter = 1
    lambda_e = 0.4
    lambda_s = 0.4
    inner = 0.925
    inner_threshold = 1.6

    tracker = ObjStruck()
    tracker.createTracker(kernel, features, filter, lambda_e, lambda_s,
                          inner, inner_threshold)
    tracker.initialize(str(seq.s_frames[0]), int(x), int(y), int(w), int(h))
    tic = time.clock()
    res = np.zeros((len(seq.s_frames),4))
    res[0] = [x,y,w,h]

    for i in range(1, len(seq.s_frames)):
        r = tracker.track(str(seq.s_frames[i]))
        res[i] = [r[0], r[1], r[2], r[3]]

    duration = time.clock() - tic

    result = dict()
    result['res'] = res
    result['type'] = 'rect'
    result['fps'] = round(seq.len / duration, 3)

    return result
