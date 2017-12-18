import sys
sys.path.append('/Users/sooda/deep/ocr/mx-textboxes/mxnet/python')
import mxnet as mx
import numpy as np
from symbol.symbol_factory import get_symbol

def merge_bn(args, auxs, conv_name, bn_prefix):
    conv_weights = args[conv_name+"_weight"].asnumpy()
    bias = args[conv_name+"_bias"].asnumpy()
    gamma = args[bn_prefix+"_gamma"].asnumpy()
    beta = args[bn_prefix+"_beta"].asnumpy()
    print conv_weights.shape, bias.shape

    mean = auxs[bn_prefix+"_moving_mean"].asnumpy()
    variance = auxs[bn_prefix+"_moving_var"].asnumpy()

    channels = conv_weights.shape[0]

    epsilon = 1e-3
    rstd = 1. / np.sqrt(variance + epsilon)

    rstd = rstd.reshape((channels, 1, 1, 1))
    gamma = gamma.reshape((channels, 1, 1, 1))
    beta = beta.reshape((channels, 1, 1, 1))
    bias = bias.reshape((channels, 1, 1, 1))
    mean = mean.reshape((channels, 1, 1, 1))

    new_weights = conv_weights * gamma * rstd
    new_bias = (bias - mean) * rstd * gamma  + beta

    new_bias = new_bias.reshape((channels,))

    print new_weights.shape, new_bias.shape
    print '----------------------'

    args[conv_name+"_weight"] = mx.nd.array(new_weights)
    args[conv_name+"_bias"] = mx.nd.array(new_bias)


if __name__ == '__main__':
    prefix  = "mobilenet_ssd"
    network = "mobilenet"
    data_shape = 300
    num_class = 20
    nms_thresh = 0.5
    force_nms = False
    nms_topk = 400
    epoch = 198

    # load model trained and saved with bn
    _, args, auxs = mx.model.load_checkpoint(prefix, epoch)

    #setup the conv layer and batchnorm layer, if you layer name with same prefix, you don't need setup this way
    conv_names = ['conv1', 'conv2_depthwise', 'conv2_pointwise', 'conv3_depthwise', 'conv3_pointwise',
                  'conv4_depthwise', 'conv4_pointwise', 'conv5_depthwise', 'conv5_pointwise',
                  'conv6_depthwise', 'conv6_pointwise', 'conv7_depthwise', 'conv7_pointwise',
                  'conv8_depthwise', 'conv8_pointwise', 'conv9_depthwise', 'conv9_pointwise',
                  'conv10_depthwise', 'conv10_pointwise', 'conv11_depthwise', 'conv11_pointwise',
                  'conv12_depthwise', 'conv12_pointwise', 'conv13_depthwise', 'conv13_pointwise',
                  'conv14_depthwise', 'conv14_pointwise'
                  ]
    bn_prefixes = ['batchnorm0', 'batchnorm1', 'batchnorm2', 'batchnorm3', 'batchnorm4',
                   'batchnorm5', 'batchnorm6', 'batchnorm7', 'batchnorm8',
                   'batchnorm9', 'batchnorm10', 'batchnorm11', 'batchnorm12',
                   'batchnorm13', 'batchnorm14', 'batchnorm15', 'batchnorm16',
                   'batchnorm17', 'batchnorm18', 'batchnorm19', 'batchnorm20',
                   'batchnorm21', 'batchnorm22', 'batchnorm23', 'batchnorm24',
                   'batchnorm25', 'batchnorm26'
                   ]
    for k, v in args.items():
        print k

    # construct no bn symbol
    nobn_sym = get_symbol(network, data_shape, num_classes=num_class, nms_thresh=nms_thresh, force_nms=force_nms,
                          nms_topk=nms_topk)

    for i in xrange(len(conv_names)):
        conv_name = conv_names[i]
        bn_prefix = bn_prefixes[i]
        merge_bn(args, auxs, conv_name, bn_prefix)

    mx.model.save_checkpoint('mergebn' , 0, nobn_sym, args, auxs)
