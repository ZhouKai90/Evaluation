import caffe
import numpy as np
import cv2
from fddb_config import config as fc


# Make sure that caffe is on the python path:
import os
os.chdir(fc.caffePath)
import sys
sys.path.insert(0, os.path.join(fc.caffePath, 'python'))
import caffe

caffe.set_device(fc.gpuID)
caffe.set_mode_gpu()
net = caffe.Net(fc.prototxt, fc.caffemodel, caffe.TEST)

count = 0
f = open(fc.detFile, 'wt')

print("start fddb test!")
for Name in open(fc.imgList, 'r'):
    Image_Path = fc.testImgPath + Name[:-1] + '.jpg'
    image = caffe.io.load_image(Image_Path)
    heigh = image.shape[0]
    width = image.shape[1]

    net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([127.5, 127.5, 127.5]))
    # transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_raw_scale('data', 255)
#     transformer.set_input_scale('data', 0.0078125)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    #print("start forword !")
    net.forward()
    #detections = net.blobs['detection_out'].data[...]
    detections = net.forward()['detection']
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3]
    det_ymin = detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]

    keep_index = np.where(det_conf >= 0)[0]
#     keep_index = [i for i, conf in enumerate(det_conf) if conf >= 0.1]
    det_conf = det_conf[keep_index]
    det_xmin = det_xmin[keep_index]
    det_ymin = det_ymin[keep_index]
    det_xmax = det_xmax[keep_index]
    det_ymax = det_ymax[keep_index]

    f.write('{:s}\n'.format(Name[:-1]))
    f.write('{:.1f}\n'.format(det_conf.shape[0]))
    for i in range(det_conf.shape[0]):
        xmin = det_xmin[i] * width
        ymin = det_ymin[i] * heigh
        xmax = det_xmax[i] * width
        ymax = det_ymax[i] * heigh
        score = det_conf[i]
        f.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.
                format(xmin, ymin, (xmax-xmin+1), (ymax-ymin+1), score))
    count += 1
    print('%d/2845' % count)
