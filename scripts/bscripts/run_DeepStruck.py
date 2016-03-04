import subprocess
import time
import numpy as np
from trackers.DeepAntrack.DeepAntrack import DeepStruck, cloneImage
import trackers.DeepAntrack.np_opencv_module as npcv
from trackers.DeepAntrack.DeepFeatureExtractor import DeepFeatureExtractor
from config import *


def run_DeepStruck(seq, rp, bSaveImage):

    x_loc = seq.init_rect[0] - 1
    y = seq.init_rect[1] - 1
    w = seq.init_rect[2]
    h = seq.init_rect[3]

    path = './results/'

    if not os.path.exists(path):
        os.makedirs(path)

    # x, y, w, h -> initial bounding box
    # seq.s_frames is a list of the images
    features = "deep"
    kernel = "linear"
    filter = 0

    dataFolder = '/Users/Ivan/Code/Tracking/DeepAntrack/data/'
    protoFile = dataFolder + 'deploy.prototxt'
    weightFile = dataFolder +'bvlc_reference_caffenet.caffemodel'
    meanFile = dataFolder + 'ilsvrc_2012_mean.npy'

    deep = DeepFeatureExtractor(protoFile_ = protoFile, weightFile_ = weightFile,
                                meanFile_ = meanFile)
    deep.load()
    deep.setComputationMode('cpu')

    print "before Struck"
    tracker = DeepStruck()
    tracker.createTracker(kernel, features, filter)
    tracker.setDisplay(0)
    tracker.initializeBefore(str(seq.s_frames[0]), int(x_loc), int(y), int(w), int(h))
    images = tracker.getImages()
    # perform feature extraction here
    for im in images:
        deep.transoformLoadedImage(im)
    batchSize = len(images)
    deep.setBatchSize(batchSize)
    print "Images for loaded..."
    x = deep.extractFeatures()

    x = cloneImage(x)
    print "First batch of the features extracted..."

    print x.shape

    tracker.initializeAfter(x)
    tic = time.clock()

    res = np.zeros((len(seq.s_frames),4))
    res[0] = [x_loc,y,w,h]
    print "Starting to track..."
    for i in range(1, len(seq.s_frames)):

        tracker.trackDetectBefore(str(seq.s_frames[i]))
        images = tracker.getImages()
        # perform feature extraction here
        batchSize = len(images)
        deep.setBatchSize(batchSize)
        for im in images:
            deep.transoformLoadedImage(im)


        x = deep.extractFeatures()
        x = cloneImage(x)
        r = tracker.trackDetectAfter(x)
        print "Index: ", i, " out of ", len(seq.s_frames)," box: ", r

        res[i] = [r[0], r[1], r[2], r[3]]

        tracker.trackUpdateBefore()

        # perform feature extraction here
        images = tracker.getImages()
        batchSize = len(images)
        deep.setBatchSize(batchSize)

        for im in images:
            deep.transoformLoadedImage(im)

        x = deep.extractFeatures()
        x = cloneImage(x)
        tracker.trackUpdateAfter(x)
    duration = time.clock() - tic

    result = dict()
    result['res'] = res
    result['type'] = 'rect'
    result['fps'] = round(seq.len / duration, 3)

    return result
