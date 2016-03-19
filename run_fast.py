import getopt
import numpy as np
from PIL import Image
import time
import itertools
from multiprocessing import *
from config import *
from scripts import *
from copy import deepcopy

def main(argv):
    t0 = time.time()
    trackers = os.listdir(TRACKER_SRC)
    #trackers = ['TEST']
    evalTypes = ['OPE', 'SRE', 'TRE']
    loadSeqs = 'ALL'
    seqs = []

    proc = 16

    try:
        opts, args = getopt.getopt(argv, "ht:e:s:p:",["tracker=","evaltype="
                                                      ,"sequence=","proc="])
    except getopt.GetoptError:
        print 'usage : run_trackers.py -t <trackers> -s <sequences>' \
            + '-e <evaltypes> -p <proc>'
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-h':
            print 'usage : run_trackers.py -t <trackers> -s <sequences>' \
                + '-e <evaltypes> -p <proc>'
            sys.exit(0)
        elif opt in ("-t", "--tracker"):
            trackers = [x.strip() for x in arg.split(',')]
            # trackers = [arg]
        elif opt in ("-s", "--sequence"):
            loadSeqs = [x.strip() for x in arg.split(',')]
        elif opt in ("-e", "--evaltype"):
            evalTypes = [x.strip() for x in arg.split(',')]
        elif opt in ("-p", "--proc"):
            proc = int(arg)
            # evalTypes = [arg]

    if SETUP_SEQ:
        print 'Setup sequences ...'
        butil.setup_seqs(loadSeqs)

    shiftTypeSet = ['left','right','up','down','topLeft','topRight',
        'bottomLeft', 'bottomRight','scale_8','scale_9','scale_11','scale_12']

    print 'Starting benchmark for {0} trackers, evalTypes : {1}'.format(
        len(trackers), evalTypes)

    for evalType in evalTypes:
        if loadSeqs == 'ALL':
            seqs = butil.load_all_seq_configs()
        else:
            for seqName in loadSeqs:
                try:
                    seq = butil.load_seq_config(seqName)
                    seqs.append(seq)
                except:
                    print 'Cannot load sequence \'{0}\''.format(seqName)
                    sys.exit(1)

        trackerResults = run_trackers(
            trackers, seqs, evalType, shiftTypeSet, proc=proc)

    t1 = time.time()

    print (t1 - t0), " Time it took to execute sequential processing"

def runParallelTrackersRun(trackers, tmpRes_path, seqs, evalType, shiftTypeSet, idxSeq):
    #print "Running the index: ", idxSeq
    s = seqs[idxSeq]
    s.len = s.endFrame - s.startFrame + 1
    s.s_frames = [None] * s.len

    for i in range(s.len):
        image_no = s.startFrame + i
        _id = s.imgFormat.format(image_no)
        s.s_frames[i] = s.path + _id

    rect_anno = s.gtRect
    numSeg = 20.0
    subSeqs, subAnno = butil.split_seq_TRE(s, numSeg, rect_anno)
    s.subAnno = subAnno
    img = Image.open(s.s_frames[0])
    (imgWidth, imgHeight) = img.size

    trackerResults = dict((t, list()) for t in trackers)
    if evalType == 'OPE':
        subS = subSeqs[0]
        subSeqs = []
        subSeqs.append(subS)

        subA = subAnno[0]
        subAnno = []
        subAnno.append(subA)

    elif evalType == 'SRE':
        subS = subSeqs[0]
        subA = subAnno[0]
        subSeqs = []
        subAnno = []
        r = subS.init_rect

        for i in range(len(shiftTypeSet)):
            subScurrent = deepcopy(subS)
            shiftType = shiftTypeSet[i]
            left = deepcopy(r)
            init_rect = butil.shift_init_BB(left, shiftType, imgH=imgHeight, imgW=imgWidth)
            subScurrent.init_rect = init_rect
            subScurrent.shiftType = shiftType
            subSeqs.append(subScurrent)
            subAnno.append(subA)
            assert subScurrent.init_rect[2] > 0
            assert subScurrent.init_rect[3] > 0

    for idxTrk in range(len(trackers)):
        t = trackers[idxTrk]
        #trackerResults[t] = list()
        if not os.path.exists(TRACKER_SRC + t):
            print '{0} does not exists'.format(t)
            sys.exit(1)
        seqResults = []
        seqLen = len(subSeqs)

        tSrc = RESULT_SRC.format(evalType) + t
        fileName = tSrc + '/{0}.json'.format(s.name)

        if os.path.exists(fileName):
            print fileName, " exists ", " ...skipping"
            continue

        for idx in range(seqLen):
            print '{0}_{1}, {2}_{3}:{4}/{5} - {6}'.format(
                idxTrk + 1, t, idxSeq + 1, s.name, idx + 1, seqLen, \
                evalType)
            rp = tmpRes_path + '_' + t + '_' + str(idx+1) + '/'
            if SAVE_IMAGE and not os.path.exists(rp):
                os.makedirs(rp)
            subS = subSeqs[idx]
            subS.name = s.name + '_' + str(idx)
            if len(subS.init_rect) == 1:
                # matlab double to python integer
                subS.init_rect = map(int, subS.init_rect[0])

            os.chdir(TRACKER_SRC + t)
            funcName = 'run_{0}(subS, rp, SAVE_IMAGE)'.format(t)
            try:
                res = eval(funcName)
            except:
                print 'failed to execute {0} : {1}'.format(
                    t, sys.exc_info())
                sys.exit(1)
            os.chdir(WORKDIR)
            res['seq_name'] = s.name
            res['len'] = subS.len
            res['annoBegin'] = subS.annoBegin
            res['startFrame'] = subS.startFrame

            if evalType == 'SRE':
                res['shiftType'] = shiftTypeSet[idx]
            seqResults.append(res)
            #end for subseqs
        evalResult, attrList = butil.calc_result_single_sequece(t, s,seqResults,evalType=evalType)
        butil.save_results_one_sequence(t, evalResult, fileName, evalType)

        trackerResults[t].append(seqResults)
    return trackerResults

def runParallelTrackersRunOne(a_b):
    return runParallelTrackersRun(*a_b)

def run_trackers(trackers, seqs, evalType, shiftTypeSet, proc=48):
    tmpRes_path = RESULT_SRC.format('tmp/{0}/'.format(evalType))
    if not os.path.exists(tmpRes_path):
        os.makedirs(tmpRes_path)

    numSeq = len(seqs)

    #
    freeze_support()
    p = Pool(proc)
    print "Number of processors being used: ", proc
    idxSequences = range(numSeq)

    for idxSeq in range(numSeq):
        s = seqs[idxSeq]
        s.len = s.endFrame - s.startFrame + 1
        s.s_frames = [None] * s.len

        for i in range(s.len):
            image_no = s.startFrame + i
            _id = s.imgFormat.format(image_no)
            s.s_frames[i] = s.path + _id

        rect_anno = s.gtRect
        numSeg = 20.0
        subSeqs, subAnno = butil.split_seq_TRE(s, numSeg, rect_anno)
        s.subAnno = subAnno

    smallSeqsNames=['Biker', 'Bird2', 'Dancer2', 'Deer', 'DragonBaby',
                    'Human8','Ironman','Jump','KiteSurf','Man','MotorRolling','Football1','Skater','Skiing','Trans']
    smallSeqs=list()
    for s in seqs:
        if s.name in smallSeqsNames:
            smallSeqs.append(s)

    idxSequences = range(len(smallSeqs))
    res = p.map(runParallelTrackersRunOne, itertools.izip(itertools.repeat(trackers),
                                                          itertools.repeat(tmpRes_path),
                                                          itertools.repeat(smallSeqs),
                                                          itertools.repeat(evalType),
                                                          itertools.repeat(shiftTypeSet), idxSequences))

if __name__ == "__main__":
    main(sys.argv[1:])
