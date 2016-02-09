import subprocess
import time
import numpy as np
from config import *


def run_DeepStruck(seq, rp, bSaveImage):
    x = seq.init_rect[0] - 1
    y = seq.init_rect[1] - 1
    w = seq.init_rect[2]
    h = seq.init_rect[3]

    path = './results/'

    if not os.path.exists(path):
        os.makedirs(path)

    # x, y, w, h -> initial bounding box
    # seq.s_frames is a list of the images

    command = map(str, ['struck.exe', 'haar', 'gaussian', '0.2', '100', '100',
                        '30', '10', bSaveImage, bSaveImage, seq.name, seq.path, seq.startFrame,
                        seq.endFrame, seq.nz, seq.ext, x, y, w, h])
    tic = time.clock()
    # subprocess.call(command)
    print "Running test tracker..."
    duration = time.clock() - tic

    result = dict()
    location = np.array([x, y, w, h])
    res = np.tile(location, (seq.len, 1))
    result['res'] = res
    result['type'] = 'rect'
    result['fps'] = round(seq.len / duration, 3)

    return result
