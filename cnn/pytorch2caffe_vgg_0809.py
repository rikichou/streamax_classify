import torch
import sys
# caffe_root = '/zyluo/caffe-master/'
# sys.path.insert(0, caffe_root + 'python')


import caffe
import numpy as np


caffemodel = 'models/epoch_16_withsoftmax.caffemodel'

protofile = 'base_model_4_classes_withsoftmax.prototxt'
pytorch_pth = 'models/epoch_16_80.5932_model_best.pth.tar'


pytorch_net = torch.load(pytorch_pth, map_location=torch.device("cpu"))['state_dict']

caffe_net = caffe.Net(protofile, caffe.TEST)

caffe_params = caffe_net.params

def save_bn2caffe(running_mean, running_var, bn_param):
    bn_param[0].data[...] = running_mean.cpu().data.numpy()
    bn_param[1].data[...] = running_var.cpu().data.numpy()
    bn_param[2].data[...] = np.array([1.0])


def save_scale2caffe(weights, biases, scale_param):
    scale_param[1].data[...] = biases.cpu().data.numpy()
    scale_param[0].data[...] = weights.cpu().data.numpy()

def pytorch2caffe():

    for (name, weights) in pytorch_net.items():

        # weights = weights + 0.002

        '''
        Convolution1
        BatchNorm1
        Scale1

        base.0.conv.weight (16, 3, 3, 3)
        base.0.bn.weight (16,)
        base.0.bn.bias (16,)
        base.0.bn.running_mean (16,)
        base.0.bn.running_var (16,)
        base.0.bn.num_batches_tracked ()
        '''
        if name == 'base.0.conv.weight':
            weights = weights.cpu().data.numpy()
            print(name)
            conv_param = caffe_params['Convolution1']
            conv_param[0].data[...] = weights
        if name == 'base.0.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.0.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.0.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.0.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm1']
            scale_params = caffe_params['Scale1']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)

            # print('conv0')

        '''
        Convolution2
        BatchNorm2
        Scale2

        base.1.conv.weight (16, 16, 3, 3)
        base.1.bn.weight (16,)
        base.1.bn.bias (16,)
        base.1.bn.running_mean (16,)
        base.1.bn.running_var (16,)
        '''
        if name == 'base.1.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution2']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.1.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.1.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.1.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.1.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm2']
            scale_params = caffe_params['Scale2']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv1DW')


        '''
        Convolution3
        BatchNorm3
        Scale3

        base.2.conv.weight (32, 16, 3, 3)
        base.2.bn.weight (32,)
        base.2.bn.bias (32,)
        base.2.bn.running_mean (32,)
        base.2.bn.running_var (32,)
        '''
        if name == 'base.2.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution3']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.2.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.2.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.2.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.2.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm3']
            scale_params = caffe_params['Scale3']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv1')


        '''
        Convolution4
        BatchNorm4
        Scale4

        base.3.conv.weight (32, 32, 3, 3)
        base.3.bn.weight (32,)
        base.3.bn.bias (32,)
        base.3.bn.running_mean (32,)
        base.3.bn.running_var (32,)
        '''
        if name == 'base.3.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution4']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.3.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.3.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.3.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.3.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm4']
            scale_params = caffe_params['Scale4']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv2DW')


        '''
        Convolution5
        BatchNorm5
        Scale5
        
        base.4.conv.weight (64, 32, 3, 3)
        base.4.bn.weight (64,)
        base.4.bn.bias (64,)
        base.4.bn.running_mean (64,)
        base.4.bn.running_var (64,)
        base.4.bn.num_batches_tracked ()
        '''
        if name == 'base.4.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution5']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.4.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.4.bn.bias':
            print(name)
            scale_biases = weights
        if name == 'base.4.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.4.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm5']
            scale_params = caffe_params['Scale5']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv2')


        '''
        Convolution6
        BatchNorm6
        Scale6

        base.5.conv.weight (64, 64, 3, 3)
        base.5.bn.weight (64,)
        base.5.bn.bias (64,)
        base.5.bn.running_mean (64,)
        base.5.bn.running_var (64,)
        '''
        if name == 'base.5.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution6']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.5.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.5.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.5.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.5.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm6']
            scale_params = caffe_params['Scale6']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv3DW')


        '''
        Convolution7
        BatchNorm7
        Scale7
        
        base.6.conv.weight (64, 64, 3, 3)
        base.6.bn.weight (64,)
        base.6.bn.bias (64,)
        base.6.bn.running_mean (64,)
        base.6.bn.running_var (64,)
        base.6.bn.num_batches_tracked ()
        '''
        if name == 'base.6.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution7']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.6.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.6.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.6.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.6.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm7']
            scale_params = caffe_params['Scale7']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv3')


        '''
        Convolution8
        BatchNorm8
        Scale8

        base.7.conv.weight (128, 64, 3, 3)
        base.7.bn.weight (128,)
        base.7.bn.bias (128,)
        base.7.bn.running_mean (128,)
        base.7.bn.running_var (128,)
        '''
        if name == 'base.7.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution8']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.7.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.7.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.7.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.7.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm8']
            scale_params = caffe_params['Scale8']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv4DW')


        '''
        Convolution9
        BatchNorm9
        Scale9

        base.8.conv.weight (128, 128, 3, 3)
        base.8.bn.weight (128,)
        base.8.bn.bias (128,)
        base.8.bn.running_mean (128,)
        base.8.bn.running_var (128,)
        '''
        if name == 'base.8.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution9']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.8.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.8.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.8.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.8.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm9']
            scale_params = caffe_params['Scale9']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv4')


        '''
        Convolution10
        BatchNorm10
        Scale10
        
        base.9.conv.weight (128, 128, 3, 3)
        base.9.bn.weight (128,)
        base.9.bn.bias (128,)
        base.9.bn.running_mean (128,)
        base.9.bn.running_var (128,)
        '''
        if name == 'base.9.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution10']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.9.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.9.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.9.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.9.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm10']
            scale_params = caffe_params['Scale10']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv5DW')


        '''
        Convolution11
        BatchNorm11
        Scale11

        base.10.conv.weight (128, 128, 3, 3)
        base.10.bn.weight (128,)
        base.10.bn.bias (128,)
        base.10.bn.running_mean (128,)
        base.10.bn.running_var (128,)
        '''
        if name == 'base.10.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution11']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.10.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.10.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.10.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.10.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm11']
            scale_params = caffe_params['Scale11']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv5')

        '''
        Convolution12
        BatchNorm12
        Scale12

        base.11.conv.weight (128, 128, 3, 3)
        base.11.bn.weight (128,)
        base.11.bn.bias (128,)
        base.11.bn.running_mean (128,)
        base.11.bn.running_var (128,)
        '''
        if name == 'base.11.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution12']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.11.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.11.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.11.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.11.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm12']
            scale_params = caffe_params['Scale12']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv6DW')


        '''
        Convolution13
        BatchNorm13
        Scale13

        base.12.conv.weight (128, 128, 3, 3)
        base.12.bn.weight (128,)
        base.12.bn.bias (128,)
        base.12.bn.running_mean (128,)
        base.12.bn.running_var (128,)
        '''
        if name == 'base.12.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution13']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.12.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.12.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.12.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.12.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm13']
            scale_params = caffe_params['Scale13']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv6')


        '''
        Convolution14
        BatchNorm14
        Scale14

        base.13.conv.weight (128, 128, 3, 3)
        base.13.bn.weight (128,)
        base.13.bn.bias (128,)
        base.13.bn.running_mean (128,)
        base.13.bn.running_var (128,)
        '''
        if name == 'base.13.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution14']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.13.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.13.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.13.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.13.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm14']
            scale_params = caffe_params['Scale14']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv7DW')


        '''
        Convolution15
        BatchNorm15
        Scale15

        base.14.conv.weight (128, 128, 3, 3)
        base.14.bn.weight (128,)
        base.14.bn.bias (128,)
        base.14.bn.running_mean (128,)
        base.14.bn.running_var (128,)
        '''
        if name == 'base.14.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution15']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.14.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.14.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.14.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.14.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm15']
            scale_params = caffe_params['Scale15']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv7')


        '''
        Convolution16
        BatchNorm16
        Scale16
        
        base.15.conv.weight (128, 128, 1, 1)
        base.15.bn.weight (128,)
        base.15.bn.bias (128,)
        base.15.bn.running_mean (128,)
        base.15.bn.running_var (128,)
        '''
        if name == 'base.15.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution16']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.15.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.15.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.15.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.15.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm16']
            scale_params = caffe_params['Scale16']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv8DW')


        '''
        Convolution17
        BatchNorm17
        Scale17

        base.16.conv.weight (128, 128, 3, 3)
        base.16.bn.weight (128,)
        base.16.bn.bias (128,)
        base.16.bn.running_mean (128,)
        base.16.bn.running_var (128,)
        base.16.bn.num_batches_tracked ()
        '''
        if name == 'base.16.conv.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['Convolution17']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'base.16.bn.weight':
            scale_weights = weights
            print(name)
        if name == 'base.16.bn.bias':
            scale_biases = weights
            print(name)
        if name == 'base.16.bn.running_mean':
            running_mean = weights
            print(name)
        if name == 'base.16.bn.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm17']
            scale_params = caffe_params['Scale17']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
            # print('conv8')

        if name == 'classifier.layers.0.weight':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['fc1']
            conv_param[0].data[...] = weights
            print(name)
        if name == 'classifier.layers.0.bias':
            weights = weights.cpu().data.numpy()
            conv_param = caffe_params['fc1']
            conv_param[1].data[...] = weights
            print(name)
    print('save caffemodel to %s' % caffemodel)
    caffe_net.save(caffemodel)


def torch_name(pytorch_net):
    for (name, value) in pytorch_net.items():
        print(name)

if __name__ == '__main__':
    print('This is main ...')
    pytorch2caffe()
    # torch_name(pytorch_net)


