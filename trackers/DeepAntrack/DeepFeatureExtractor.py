import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe

class DeepFeatureExtractor(object):

    protoFile= None
    weightFile= None
    meanFile = None
    batchSize = 128
    transformer = None
    net = None
    imageNumberInQueue = 0

    def __init__(self, protoFile_="", weightFile_ ="", meanFile_=""):
        """Initialize the feature extraction method

        :param protoFile_: Path to network definition file (file ending with *.prototxt)
        :param weightFile_: Path to the weights of the network (file ending with *.caffemodel)
        :param meanFile_: Mean file from images computed over imagenet dataset
        """
        self.protoFile = protoFile_
        self.weightFile = weightFile_
        self.meanFile = meanFile_

    def setBatchSize(self, batchSize_):
        self.batchSize = batchSize_
        self.net.blobs['data'].reshape( self.batchSize, 3, 227, 227)


    def load(self):
        """ Load the network by reading weightfile and protofile from the
        disk using variables protoFile, weightFile and meanFile
        """

        self.net = caffe.Net( self.protoFile, self.weightFile , caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data',np.load(self.meanFile).mean(1).mean(1)) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2,1,0))
        self.net.blobs['data'].reshape( self.batchSize, 3, 227, 227)

    def transform(self, imageFile):
        """Perform image transformation to make the data right size and shape
        for the network

        :param imageFile: Image path
        """
        self.net.blobs['data'].data[self.imageNumberInQueue] = self.transformer.preprocess('data', caffe.io.load_image(imageFile))
        self.imageNumberInQueue = self.imageNumberInQueue + 1

    def transoformLoadedImage(self, image):
        """Perform image transformation to make the data right size and shape
        for the network

        :param imageFile: Image path
        """

        self.net.blobs['data'].data[self.imageNumberInQueue] = self.transformer.preprocess('data', image)
        self.imageNumberInQueue = self.imageNumberInQueue + 1


    def setComputationMode(self, cpuORgpu='cpu'):
        """Set the mode of computation ('cpu' OR 'gpu')

        :param cpuORgpu: mode
        """

        if cpuORgpu == 'cpu':
            caffe.set_mode_cpu()
        elif cpuORgpu == 'gpu':
            caffe.set_mode_gpu()
        else:
            raise ValueError('Unknown computation mode')

    def extractFeatures(self, layer='fc7'):
        """ Extract features from the layer

        :param layer: Name of the layer to take the features from
        :returns: Numpy array
        :rtype: ndarray

        """
        self.imageNumberInQueue = 0
        out = self.net.forward()
        return self.net.blobs[layer].data.copy()

def main():
    dataFolder = '/Users/Ivan/Code/Tracking/DeepAntrack/data/'
    dataFolder = '/udrive/student/ibogun2010/Research/Code/DeepAntrack/data/';
    protoFile = dataFolder + 'deploy.prototxt'
    weightFile = dataFolder +'bvlc_reference_caffenet.caffemodel'
    meanFile = dataFolder + 'ilsvrc_2012_mean.npy'

    deep = DeepFeatureExtractor(protoFile_ = protoFile, weightFile_ = weightFile,
                                meanFile_ = meanFile)
    deep.load()
    deep.setComputationMode('gpu')

    imgFolder = '/udrive/student/ibogun2010/Research/Code/Antrack/python/visual-tracking-benchmark/data/Doll/img/';
    
    #imageFile = '/Users/Ivan/Code/deep/caffe/examples/images/cat.jpg'
    miniBatch = 0
    maxBatch = 743
    deep.setBatchSize(maxBatch)
    for j in range(0, 50):
        for i in range(1000,3873):
            imageFile = imgFolder +str(i)+'.jpg'
            print j, i
            deep.transform(imageFile)
            miniBatch = miniBatch + 1
            if miniBatch >= maxBatch:
                features = deep.extractFeatures(layer='fc7')
                miniBatch = 0
    #print features[0]
    #print type(features)
    #print features[0].shape



if  __name__ =='__main__':main()
