import subprocess
import time
import numpy as np

from trackers.Antrack.antrack import MStruck
from config import *


def run_M128L0_1Deep(seq, rp, bSaveImage):
    x = seq.init_rect[0] - 1
    y = seq.init_rect[1] - 1
    w = seq.init_rect[2]
    h = seq.init_rect[3]

    path = './results/'

    if not os.path.exists(path):
        os.makedirs(path)

    # x, y, w, h -> initial bounding box
    # seq.s_frames is a list of the images

    top_features = "deep"
    top_kernel = "linear"
    filter=0
    dis_features = "hogANDhist"
    dis_kernel = "linear"
    features = "hogANDhist"
    kernel = "int"

    tracker = MStruck()

    dataFolder = '/udrive/student/ibogun2010/Research/Code/DeepAntrack/data/'

    M = 128;
    l = 0.1;
    B_top = 100;
    B_bottom = 100;
    tracker.deepFeatureParams(dataFolder)
    tracker.createTracker(kernel, features, filter,
                          dis_features, dis_kernel,
                          top_features, top_kernel)


    tracker.setM(M);
    tracker.setLambda(l);
    tracker.setTopBudget(B_top);
    tracker.setBottomBudget(B_bottom);
    tracker.setDisplay(0)
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
