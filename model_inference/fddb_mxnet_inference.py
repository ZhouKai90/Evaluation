import numpy as np
import cv2
import mxnet as mx
from fddb_config import config as fc

prefix = "ssdface"
epoch = 0
symbol, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

count = 0
f = open(fc.detFile, 'wt')
ctx = mx.cpu()
for Name in open(fc.imgList, 'r'):
    Image_Path = fc.testImgPath + Name[:-1] + '.jpg'
    image = cv2.imread(Image_Path)
    heigh = image.shape[0]
    width = image.shape[1]
    input_array = np.zeros(shape=(1,3,image.shape[0], image.shape[1]))
    input_array[0,0,:,:] = image[:,:,2]
    input_array[0,1,:,:] = image[:,:,1]
    input_array[0, 2, :, :] = image[:, :, 0]
    arg_params['data'] = mx.nd.array(input_array, ctx)
    exector = symbol.bind(ctx, arg_params, args_grad=None,
                          grad_req="null", aux_states=aux_params)
    exector.forward(is_train=False)
    exector.outputs[0].wait_to_read()
    output = exector.outputs[0].asnumpy()
    # print("output.shape() = ", type(output), output.shape)
    detections = output[0, :, :]

    det_id   = detections[:, 0]
    det_conf = detections[:, 1]
    det_xmin = detections[:, 2]
    det_ymin = detections[:, 3]
    det_xmax = detections[:, 4]
    det_ymax = detections[:, 5]

    # print(np.where(det_id >=0)[0])
    keep_index = np.where(det_id >= 0)[0]
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
