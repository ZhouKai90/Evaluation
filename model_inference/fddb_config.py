# -*- coding: UTF-8 -*-

from easydict import EasyDict as edict

config = edict()

config.caffePath = '/kyle/workspace/framework/caffe/caffe_ssd'
config.rootPath = '/kyle/workspace/project/tools/evaluation/'
config.gpuID = 1
config.prototxt = config.rootPath + 'model_inference/model/deploy_ssd_inceptionv3_512-symbol.prototxt'
config.caffemodel = config.rootPath + 'model_inference/model/deploy_ssd_inceptionv3_512-0000.caffemodel'

config.testImgPath = '/kyle/workspace/dataset/public/95_FDDB/'

config.detFile = config.rootPath + 'model_inference/fddb_det.lst'
config.imgList = config.rootPath + 'model_inference/fddb_img_list.txt'


