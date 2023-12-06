#（1）继承并写新的量化算子；

import math
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .function import *

from .ops import *
import genotypes as gt
#added, for QMixedOp
QOPS = {
    'none': lambda C, stride, affine: QZero(stride, qi=False, qo=True, num_bits=8),
    'avg_pool_3x3': lambda C, stride, affine: \
        QPoolBN('avg', C, 3, stride, 1, affine=affine, qi=False, qo=True, num_bits=8),
    'max_pool_3x3': lambda C, stride, affine: \
        QPoolBN('max', C, 3, stride, 1, affine=affine, qi=False, qo=True, num_bits=8),
    'skip_connect': lambda C, stride, affine: \
        QIdentity(qi=False, qo=True, num_bits=8) if stride == 1 \
            else QFactorizedReduce(C, C, affine=affine, qi=False, qo=True, num_bits=8),
    'sep_conv_3x3': lambda C, stride, affine: \
        QSepConv(C, C, 3, stride, 1, affine=affine, qi=False, qo=True, num_bits=8),
    'sep_conv_5x5': lambda C, stride, affine: \
        QSepConv(C, C, 5, stride, 2, affine=affine, qi=False, qo=True, num_bits=8),
    'sep_conv_7x7': lambda C, stride, affine: \
        QSepConv(C, C, 7, stride, 3, affine=affine, qi=False, qo=True, num_bits=8),
    'dil_conv_3x3': lambda C, stride, affine: \
        QDilConv(C, C, 3, stride, 2, 2, affine=affine, qi=False, qo=True, num_bits=8), # 5x5
    'dil_conv_5x5': lambda C, stride, affine: \
        QDilConv(C, C, 5, stride, 4, 2, affine=affine, qi=False, qo=True, num_bits=8), # 9x9
    'conv_7x1_1x7': lambda C, stride, affine: \
        QFacConv(C, C, 7, stride, 3, affine=affine, qi=False, qo=True, num_bits=8)
}


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    if min_val == max_val:#处理最大值和最小值相等的情况
        #bug fixed, scale是一个buffer下的tensor,不能用浮点数赋值。
        # scale = 1.0
        scale = torch.tensor(1.0)
    else:
        scale = (max_val - min_val) / (qmax - qmin)

    zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min_val.device)
    elif zero_point > qmax:
        # zero_point = qmax
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max_val.device)

    zero_point.round_()

    return scale, zero_point


def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1.

    #test
    # if scale.shape == torch.Size([0]):
    #     print(zero_point.shape, x.shape)
    #     print(zero_point, x, scale)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()

    return q_x


def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x - zero_point)


def search(M):
    P = 7000
    n = 1
    while True:
        Mo = int(round(2 ** n * M))
        # Mo 
        approx_result = Mo * P >> n
        result = int(round(M * P))
        error = approx_result - result

        print("n=%d, Mo=%f, approx=%d, result=%d, error=%f" % \
              (n, Mo, approx_result, result, error))

        if math.fabs(error) < 1e-9 or n >= 22:
            return Mo, n
        n += 1


class QParam(nn.Module):

    def __init__(self, num_bits=8):
        super(QParam, self).__init__()
        self.num_bits = num_bits
        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)
        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)
        self.register_buffer('scale', scale)
        self.register_buffer('zero_point', zero_point)
        self.register_buffer('min', min)
        self.register_buffer('max', max)

    def update(self, tensor):
        #暂时不使用，因为这里不需要QSoftmax
        #added, 为了在量化softmax中的exp时，量化ln2，从而计算z
        #self.max.data = torch.log(torch.tensor([2]))
        #added

        if self.max.nelement() == 0 or self.max.data < tensor.max().data:
            self.max.data = tensor.max().data
        self.max.clamp_(min=0)

        if self.min.nelement() == 0 or self.min.data > tensor.min().data:
            self.min.data = tensor.min().data
        self.min.clamp_(max=0)

        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max, self.num_bits)

    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, num_bits=self.num_bits)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        key_names = ['scale', 'zero_point', 'min', 'max']
        for key in key_names:
            value = getattr(self, key)
            value.data = state_dict[prefix + key].data
            state_dict.pop(prefix + key)

    def __str__(self):
        info = 'scale: %.10f ' % self.scale
        info += 'zp: %d ' % self.zero_point
        info += 'min: %.6f ' % self.min
        info += 'max: %.6f' % self.max
        return info

class QModule(nn.Module):

    def __init__(self, qi=False, qo=True, num_bits=8):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(num_bits=num_bits)
        if qo:
            self.qo = QParam(num_bits=num_bits)
        
        self.qi_flag = qi

    def freeze(self):
        pass

    #added, 为了解决历史qi导致NaN出现的bug
    def cleanQi(self):
        #反量化量化后的weight和bias的
        for name, module in self.named_children():
            if 'module' in name:
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.data = module.weight.data + self.qw.zero_point
                    module.weight.data = self.qw.dequantize_tensor(module.weight.data)

                if hasattr(module, 'bias') and module.bias is not None:
                    # if hasattr(self, 'qi') is False or hasattr(self, 'qw') is False:
                        # print(self)
                    module.bias.data = dequantize_tensor(module.bias, scale=self.qi.scale*self.qw.scale,
                                                         zero_point=torch.tensor(0.))

        if self.qi_flag is False and hasattr(self, 'qi'):
            del self.qi
        elif self.qi_flag is True and hasattr(self, 'qi'):
            self.qi.min = torch.tensor([], requires_grad=False)
            self.qi.max = torch.tensor([], requires_grad=False)
        if hasattr(self, 'qw'):
            self.qw.min = torch.tensor([], requires_grad=False)
            self.qw.max = torch.tensor([], requires_grad=False)
        if hasattr(self, 'qo'):
            self.qo.min = torch.tensor([], requires_grad=False)
            self.qo.max = torch.tensor([], requires_grad=False)
        
        #特殊处理QMixedOp
        if hasattr(self, 'w'):
            self.w.data = self.w.data + self.qw.zero_point
            self.w.data = self.qw.dequantize_tensor(self.w.data)

        if isinstance(self, QMixedOp):
            del self.qi_list
        
        #test
        # print(self.__class__.__name__, ' is cleaned.')
        # if(self.__class__.__name__ == 'QSearchCell'):
        # print('\n----test start')
        # preModule = 'noModule'
        # for name, module in self.named_modules():
        #     if 'q' in name and preModule not in name and isinstance(module, (QModule, QModule2)):
        #         print(name)
        #         preModule = name
        #         # print(module)
        # print('----test end\n')
            
        if len(list(self.children())) != 0:
            preModule = 'noModule'
            for name, module in self.named_modules():
                if 'q' in name and preModule not in name and isinstance(module, (QModule, QModule2)):
                    preModule = name
                    # print(name, 'is prepared to be cleaned:')
                    # print(module)
                    module.cleanQi()
        #test
        # print(self.__class__.__name__, ' has no children.')


    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented.') 
    
#added for opAdd & opConcat,多操作数形式的量化参数Module
#nn.ModuleList 和 list有什么区别？
class QModule2(nn.Module):

    def __init__(self, qi_list=False, qo=True, num_bits=8):
        super(QModule2, self).__init__()
        if qi_list:
            self.qi_list = []
            for qi in qi_list:
                if qi:
                    self.qi_list.append(QParam(num_bits=num_bits))
        
        if qo:
            self.qo = QParam(num_bits=num_bits)
    
        
        self.qi_flag = qi_list

    def freeze(self):
        pass

    #added, 为了解决历史qi导致NaN出现的bug
    def cleanQi(self):
        if self.qi_flag is False and hasattr(self, 'qi_list'):
            del self.qi_list
        elif self.qi_flag is True and hasattr(self, 'qi_list'):
            for qi in self.qi_list:
                qi.min = torch.tensor([], requires_grad=False)
                qi.max = torch.tensor([], requires_grad=False)
        if hasattr(self, 'qw'):
            self.qw.min = torch.tensor([], requires_grad=False)
            self.qw.max = torch.tensor([], requires_grad=False)
        if hasattr(self, 'qo'):
            self.qo.min = torch.tensor([], requires_grad=False)
            self.qo.max = torch.tensor([], requires_grad=False)
        
        # test, 当QSearchCell中不写cleanQi函数，就会cleanQi两次，所以要写
        # print(self.__class__.__name__, ' is cleaned.')
        # if(self.__class__.__name__ == 'QSearchCell'):
        # print('\n----test start')
        # preModule = 'noModule'
        # for name, module in self.named_modules():
        #     if 'q' in name and preModule not in name and isinstance(module, (QModule, QModule2)):
        #         print(name)
        #         preModule = name
        #         # print(module)
        # print('----test end\n')
        if len(list(self.children())) != 0:
            preModule = 'noModule'
            for name, module in self.named_modules():
                if 'q' in name and preModule not in name and isinstance(module, (QModule, QModule2)):
                    preModule = name
                    # print(name, 'is prepared to be cleaned:')
                    # print(module)
                    module.cleanQi()

    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented.') 
    
class QMaxPooling2d(QModule):

    def __init__(self, kernel_size=3, stride=1, padding=0, qi=False, num_bits=None):
        super(QMaxPooling2d, self).__init__(qi=qi, num_bits=num_bits)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

        return x

    def quantize_inference(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)  

    def fake_quantize_inference(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)  

# self_defined, added
class QAvgPool2d(QModule):

    def __init__(self, kernel_size=3, stride=1, padding=0, count_include_pad=True, qi=False, num_bits=None):
        super(QAvgPool2d, self).__init__(qi=qi, num_bits=num_bits)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.count_include_pad = count_include_pad

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding, \
                         count_include_pad=self.count_include_pad)

        return x

    def quantize_inference(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
    
    def fake_quantize_inference(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
    


# self_defined, added, Batchnorm2d
def params_get(X, gamma, beta, moving_mean, moving_var, eps=1e-5, momentum=0.9, affine=True):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    #if not torch.is_grad_enabled():
    if not affine:
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        #X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        std = torch.sqrt(moving_var + eps)
        mean_ = moving_mean
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        #X_hat = (X - mean) / torch.sqrt(var + eps)
        std = torch.sqrt(var + eps)
        mean_ = mean
        # 更新移动平均的均值和方差
        #print(moving_mean.device, mean.device, X.device)
        #为啥moving_mean在cpu上,而X在cuda上？
        # 因为moving_mean是在自定义BN层的forward中才移动到和x相同设备上，而调用本函数的某些情况下没移动
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    
    weight = gamma / std
    bias = -weight * mean_ + beta
    return moving_mean.data, moving_var.data, weight, bias

def params_inf(gamma, beta, moving_mean, moving_var, eps=1e-5, affine=False):
    assert affine==False
    std = torch.sqrt(moving_var + eps)
    weight = gamma / std
    bias = -weight * moving_mean + beta
    return weight, bias

def batch_norm(X, weight, bias):
    Y = weight * X + bias  # 缩放和移位
    return Y

# 改版bn层通过了损失测试！
class BatchNorm2d(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    # 此处不考虑完全连接层的情况
    def __init__(self, num_features, num_dims=4, affine=True, eps=1e-5, momentum=0.9):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
        # 记录通道数
        # self.channel = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var,
        # 以及计算出来的适合量化的weight和bias
        #print('------moving_mean is on ', self.moving_mean.device)
        self.moving_mean, self.moving_var, self.weight, self.bias = params_get(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=self.eps, momentum=self.momentum, affine=self.affine)
        Y = batch_norm(X, self.weight, self.bias)
        return Y

class QBatchNorm2d(QModule):
    def __init__(self, bn_module, qi=False, qo=True, num_bits=8):
        super(QBatchNorm2d, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.bn_module = bn_module
        self.qw = QParam(num_bits=num_bits)
        self.qb = QParam(num_bits=32)
        self.register_buffer('M', torch.tensor([], requires_grad=False))  # 将M注册为buffer

    def forward(self, x):
        if self.bn_module.moving_mean.device != x.device:
            self.bn_module.moving_mean = self.bn_module.moving_mean.to(x.device)
            self.bn_module.moving_var = self.bn_module.moving_var.to(x.device)
        #print(self.bn_module.moving_mean.device, x.device, '------')
        self.bn_module.moving_mean, self.bn_module.moving_var, self.bn_module.weight, self.bn_module.bias = params_get(
            x, self.bn_module.gamma, self.bn_module.beta, self.bn_module.moving_mean,
            self.bn_module.moving_var, eps=self.bn_module.eps, momentum=self.bn_module.momentum)

        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.bn_module.weight.data)

        # Ques: 注意：此处我没对bias进行量化反量化，这可能是导致量化推理效果差的原因？

        # Ques: qi是在推理过程中调用freeze后才有的，这意味着训练时x没经过量化反量化？
        # ANs: 已经经过量化反量化，因为上一层的输出就是本层的输入，而上一层的输出已经量化反量化
        x = batch_norm(x, FakeQuantize.apply(self.bn_module.weight, self.qw), 
                       self.bn_module.bias)

        #test batchnorm's quant loss
        # x_old = copy.copy(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        #test batchnorm's quant loss
        # loss_batch = torch.mean((x_old-x)**2)
        # print('loss_batch is ', loss_batch)

        return x

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data

        #added, 在推理时候，需要再更新一下weight和bias，因为训练和推理使用的bn参数不同
        self.bn_module.weight, self.bn_module.bias = params_inf(
            self.bn_module.gamma, self.bn_module.beta, self.bn_module.moving_mean,
            self.bn_module.moving_var, eps=1e-5)
        self.bn_module.weight.data = self.qw.quantize_tensor(self.bn_module.weight.data)
        self.bn_module.weight.data = self.bn_module.weight.data - self.qw.zero_point

        if self.bn_module.bias is not None:
            self.bn_module.bias.data = quantize_tensor(self.bn_module.bias, scale=self.qi.scale * self.qw.scale,
                                                     zero_point=torch.tensor(0.), num_bits=32, signed=True)

    def quantize_inference(self, x):
        # 推理时不经过以下操作，acc50
        # return x

        x = x - self.qi.zero_point
        # 成功了！问题在于执行self.bn_module时weight和bias会被覆盖为浮点数，推理前的freeze操作白费了
        # x = self.bn_module(x)
        x = batch_norm(x, self.bn_module.weight, self.bn_module.bias)
        x = self.M * x
        x.round_() 
        x = x + self.qo.zero_point        
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x
    
    def fake_quantize_inference(self, x):
        if self.bn_module.moving_mean.device != x.device:
            self.bn_module.moving_mean = self.bn_module.moving_mean.to(x.device)
            self.bn_module.moving_var = self.bn_module.moving_var.to(x.device)
        self.bn_module.moving_mean, self.bn_module.moving_var, self.bn_module.weight, self.bn_module.bias = params_get(
            x, self.bn_module.gamma, self.bn_module.beta, self.bn_module.moving_mean,
            self.bn_module.moving_var, eps=self.bn_module.eps, momentum=self.bn_module.momentum)

        if hasattr(self, 'qi'):
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.bn_module.weight.data)

        x = batch_norm(x, FakeQuantize.apply(self.bn_module.weight, self.qw), 
                       self.bn_module.bias)

        if hasattr(self, 'qo'):
            x = FakeQuantize.apply(x, self.qo)

        return x
    

class QReLU(QModule):

    def __init__(self, qi=False, num_bits=None):
        super(QReLU, self).__init__(qi=qi, num_bits=num_bits)

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.relu(x)

        return x

    def quantize_inference(self, x):
        x = x.clone()
        x[x < self.qi.zero_point] = self.qi.zero_point
        return x
    
    def fake_quantize_inference(self, x):
        if hasattr(self, 'qi'):
            x = FakeQuantize.apply(x, self.qi)

        x = F.relu(x)
        return x

class QConv2d(QModule):
    # 如果此处的qi为真(默认)，则在生成对象时会初始化一个self.qi；如果为假，则不初始化，会在调用freeze函数时被赋值。(见QModule);
    def __init__(self, conv_module, qi=False, qo=True, num_bits=8):   # 此处qi和qo是Bool类
        super(QConv2d, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module
        self.qw = QParam(num_bits=num_bits)
        self.register_buffer('M', torch.tensor([], requires_grad=False))  # 将M注册为buffer

    #  量化算子的freeze干什么：根据量化训练完成后确定下来的量化参数，对每一层的w和x进行量化
    def freeze(self, qi=None, qo=None):  # 此处qi和qo是QParam类
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data

        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point

        #test
        # print(self.conv_module.bias)
        #bug, conv的bias可能为空
        if self.conv_module.bias is not None:
            self.conv_module.bias.data = quantize_tensor(self.conv_module.bias.data, scale=self.qi.scale * self.qw.scale,
                                                     zero_point=0, num_bits=32, signed=True)

    def forward(self, x):  # !在量化训练时候用
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.conv_module.weight.data)

        x = F.conv2d(x, FakeQuantize.apply(self.conv_module.weight, self.qw), self.conv_module.bias,
                     stride=self.conv_module.stride,
                     padding=self.conv_module.padding, dilation=self.conv_module.dilation,
                     groups=self.conv_module.groups)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    #  量化推理inference干什么：在freeze之后即确定量化后的w和x后，以定点运算方式算每一层，此处是对QConv2d这一层的推理
    def quantize_inference(self, x):  # !在推理时候用
        x = x - self.qi.zero_point
        x = self.conv_module(x)  # 此处调用了原conv2d的forward, 在推理时使用(注意:此处x是量化后的x，得到的也是定点x)
        x = self.M * x
        x.round_()
        x = x + self.qo.zero_point
        x.clamp_(0., 2. ** self.num_bits - 1.).round_()
        return x
    
    def fake_quantize_inference(self, x):  # !在量化训练时候用
        if hasattr(self, 'qi'):
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.conv_module.weight.data)

        x = F.conv2d(x, FakeQuantize.apply(self.conv_module.weight, self.qw), self.conv_module.bias,
                     stride=self.conv_module.stride,
                     padding=self.conv_module.padding, dilation=self.conv_module.dilation,
                     groups=self.conv_module.groups)

        if hasattr(self, 'qo'):
            x = FakeQuantize.apply(x, self.qo)
        return x

# self_defined, added, QIdentity
# Op Identity() has neither weights nor bias!!!
# So in forward(): q&dq x -> do nothing -> q&dq x
class QIdentity(QModule):
    def __init__(self, qi=False, qo=True, num_bits=8):   
        super(QIdentity, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.identity_module = Identity()

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        # the reason why don't use the nn.Identity is the prob that 
        # it may ruin the STE mechanism(maybe not)
        x = self.identity_module(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x):  
        x = self.identity_module(x)
        return x
    
    def fake_quantize_inference(self, x):
        if hasattr(self, 'qi'):
            x = FakeQuantize.apply(x, self.qi)

        x = self.identity_module(x)

        if hasattr(self, 'qo'):
            x = FakeQuantize.apply(x, self.qo)
        return x
    
# self_defined, added, QZero
# Op Zero() has neither weights nor bias!!!
# So in forward(): q&dq x -> do nothing -> q&dq x
# def Zero(x, stride=1):
#     if stride == 1:
#             return x * 0.

#     # re-sizing by stride
#     return x[:, :, ::stride, ::stride] * 0.

# 需要考虑情况，原来range[1,10], 后来全化为0了, range[0, 10]
# 其实zero比较特殊，经过了zero操作之后，量化推理时不需要考虑qi，只需要保证qo能量化反量化0就行
class QZero(QModule):
    def __init__(self, stride, qi=False, qo=True, num_bits=8):   
        super(QZero, self).__init__(qi=qi, qo=True, num_bits=num_bits)
        self.num_bits = num_bits
        self.zero_module = Zero(stride=stride)

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        
        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = self.zero_module(x)
        #test
        # print('x = ', type(x))

        if hasattr(self, 'qo'):
            self.qo.update(x)# a trick，可能bug, fixed
            x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x):
        # bug, 化零算子量化版区别于原版，即qi.zero!=zero  
        # x = self.zero_module(x)
        if self.zero_module.stride == 1:
            # return x * 0.
            x.fill_(self.qi.zero_point)
        else:
            # return x[:, :, ::self.stride, ::self.stride] * 0.
            x[:, :, ::self.zero_module.stride, ::self.zero_module.stride] = self.qi.zero_point
            x = x[:, :, ::self.zero_module.stride, ::self.zero_module.stride]
        return x
    
    def fake_quantize_inference(self, x):
        if hasattr(self, 'qi'):
            x = FakeQuantize.apply(x, self.qi)

        x = self.zero_module(x)

        if hasattr(self, 'qo'):
            x = FakeQuantize.apply(x, self.qo)

        return x

# self_defined, added, QAdaptiveAvgPool2d
class QAdaptiveAvgPool2d(QModule):

    def __init__(self, output_size = 3, qi=False, num_bits=None):
        super(QAdaptiveAvgPool2d, self).__init__(qi=qi, num_bits=num_bits)
        self.output_size = output_size

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.adaptive_avg_pool2d(x, self.output_size)

        return x

    def quantize_inference(self, x):
        #bug, 直接return会得到浮点数，因为avg_pool中，每个元素是平均数，即可能是浮点数
        # return F.adaptive_avg_pool2d(x, self.output_size)
        x = F.adaptive_avg_pool2d(x, self.output_size)
        x.round_()
        return x
    
    def fake_quantize_inference(self, x):
        if hasattr(self, 'qi'):
            x = FakeQuantize.apply(x, self.qi)

        x = F.adaptive_avg_pool2d(x, self.output_size)
        return x


class QLinear(QModule):
    def __init__(self, fc_module, qi=False, qo=True, num_bits=8):
        super(QLinear, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.fc_module = fc_module
        self.qw = QParam(num_bits=num_bits)
        self.register_buffer('M', torch.tensor([], requires_grad=False))  # 将M注册为buffer

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data

        self.fc_module.weight.data = self.qw.quantize_tensor(self.fc_module.weight.data)
        self.fc_module.weight.data = self.fc_module.weight.data - self.qw.zero_point
        #test
        # print('qi.scale and qw.scale is: ', self.qi.scale, self.qw.scale)
        if self.fc_module.bias is not None:
            self.fc_module.bias.data = quantize_tensor(self.fc_module.bias.data, scale=self.qi.scale * self.qw.scale,
                                                   zero_point=torch.tensor(0.), num_bits=32, signed=True)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.fc_module.weight.data)

        x = F.linear(x, FakeQuantize.apply(self.fc_module.weight, self.qw), self.fc_module.bias)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.fc_module(x)
        x = self.M * x
        x.round_()
        x = x + self.qo.zero_point
        x.clamp_(0., 2. ** self.num_bits - 1.).round_()
        return x
    
    def fake_quantize_inference(self, x):
        if hasattr(self, 'qi'):
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.fc_module.weight.data)

        x = F.linear(x, FakeQuantize.apply(self.fc_module.weight, self.qw), self.fc_module.bias)

        if hasattr(self, 'qo'):
            x = FakeQuantize.apply(x, self.qo)
        return x
    
    """
    classification:
    the upper ones are simple quantized ops;
    and the following ones are fused ones
    """
# how to define QPoolBN?
# is it really required? yes, cause it is part of mix ops
# essentially the same to how to define QBN!
#注意：此处bn量化算子不定义channel，需再次在bn算子处定义channel

#added, 统一max和avg的函数规范,返pool操作后的结果
def pool_function(x, pool_type, kernel_size, stride, padding):
    if pool_type.lower() == 'max':
        x = F.max_pool2d(x, kernel_size, stride, padding)
    elif pool_type.lower() == 'avg':
        x = F.avg_pool2d(x, kernel_size, stride, padding, count_include_pad=False)
    else:
        raise ValueError()
    return x

#added, QPoolBN, 复合算子1
#这里的逻辑很奇怪，k,s,p参数应该赋值给self.pool而不是self, fixed
class QPoolBN(QModule):
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True, qi=False, qo=True, num_bits=8):
        super(QPoolBN, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        if pool_type.lower() == 'max':
            self.qPool = QMaxPooling2d(kernel_size, stride, padding, \
                                       qi=False, num_bits=num_bits)
        elif pool_type.lower() == 'avg':
            self.qPool = QAvgPool2d(kernel_size, stride, padding, count_include_pad=False, \
                                    qi=False, num_bits=num_bits)
        else:
            raise ValueError()
        bn_module = BatchNorm2d(C, affine=affine)
        self.qBn = QBatchNorm2d(bn_module, qi=False, qo=True, num_bits=num_bits)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = self.qPool(x)
        x = self.qBn(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qi = qo

        self.qPool.freeze(self.qi)
        self.qBn.freeze(self.qi)

    def quantize_inference(self, x):
        x = self.qPool.quantize_inference(x)
        x = self.qBn.quantize_inference(x)
        return x
    
    def fake_quantize_inference(self, x):
        x = self.qPool.fake_quantize_inference(x)
        x = self.qBn.fake_quantize_inference(x)
        return x

#added, QStdConv,that is relu->conv->bn
#注意，此处传的应该是nn.Module中的Bn层，区别于QPoolBN中传输的自定义的Module中的Bn层
#注意，推理时使用的clamp不知道能不能起到relu的作用
class QStdConv(QModule):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, qi=False, qo=True, num_bits=8):
        super(QStdConv, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        conv_module = nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False)
        bn_module = BatchNorm2d(C_out, affine=affine)

        self.qrelu = QReLU()
        self.qconv = QConv2d(conv_module, qi=False, qo=True, num_bits=num_bits)
        self.qbn = QBatchNorm2d(bn_module, qi=False, qo=True, num_bits=num_bits)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = self.qrelu(x)
        x = self.qconv(x)
        x = self.qbn(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)
        return x

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        
        self.qrelu.freeze(self.qi)
        self.qconv.freeze(self.qi)
        self.qbn.freeze(self.qconv.qo)

    def quantize_inference(self, x):
        qx = self.qrelu.quantize_inference(x)
        qx = self.qconv.quantize_inference(qx)
        qx = self.qbn.quantize_inference(qx)
        return qx
    
    def fake_quantize_inference(self, x):
        qx = self.qrelu.fake_quantize_inference(x)
        qx = self.qconv.fake_quantize_inference(qx)
        qx = self.qbn.fake_quantize_inference(qx)
        return qx

#added, QFacConv, QDilConv, that is relu->conv->conv->bn, 复合算子2
#注意，此处传的应该是nn.Module中的Bn层，区别于QPoolBN中传输的自定义的Module中的Bn层
#注意，推理时使用的clamp不知道能不能起到relu的作用
#注意，当前算子是真正意义上的多算子(包含多个量化算子，不计算量化参数，只负责传递)

#难点，如何确认量化参数？将其看成两部分，分别是relu->conv和conv->bn
#疑问，为什么不看成四部分呢？先实现它，因为这样代码最好看。
#疑问2，四部分和两部分的区别是什么呢？
class QFacConv(QModule):
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True, qi=False, qo=True, num_bits=8):
        super(QFacConv, self).__init__(qi=qi, qo=qo, num_bits=num_bits)

        conv1 = nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False)
        conv2 = nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False)
        bn_module = BatchNorm2d(C_out, affine=affine)

        self.num_bits = num_bits
        self.qrelu = QReLU()
        self.qconv1 = QConv2d(conv1, qi=False, qo=True, num_bits=num_bits)
        self.qconv2 = QConv2d(conv2, qi=False, qo=True, num_bits=num_bits)
        self.qbn = QBatchNorm2d(bn_module, qi=False, qo=True, num_bits=num_bits)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = self.qrelu(x)
        x = self.qconv1(x)
        x = self.qconv2(x)
        x = self.qbn(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)
        return x

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        
        self.qrelu.freeze(self.qi)
        self.qconv1.freeze(self.qi)
        self.qconv2.freeze(self.qconv1.qo)
        self.qbn.freeze(self.qconv2.qo)

    def quantize_inference(self, x):
        qx = self.qrelu.quantize_inference(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qbn.quantize_inference(qx)
        return qx
    
    def fake_quantize_inference(self, x):
        qx = self.qrelu.fake_quantize_inference(x)
        qx = self.qconv1.fake_quantize_inference(qx)
        qx = self.qconv2.fake_quantize_inference(qx)
        qx = self.qbn.fake_quantize_inference(qx)
        return qx

#added, QDilConv, 复合算子3
class QDilConv(QModule):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, qi=False, qo=True, num_bits=8):
        super(QDilConv, self).__init__(qi=qi, qo=qo, num_bits=num_bits)

        conv1 = nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False)
        conv2 = nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
        bn_module = BatchNorm2d(C_out, affine=affine)

        self.num_bits = num_bits
        self.qrelu = QReLU()
        self.qconv1 = QConv2d(conv1, qi=False, qo=True, num_bits=num_bits)
        self.qconv2 = QConv2d(conv2, qi=False, qo=True, num_bits=num_bits)
        self.qbn = QBatchNorm2d(bn_module, qi=False, qo=True, num_bits=num_bits)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = self.qrelu(x)
        x = self.qconv1(x)
        x = self.qconv2(x)
        x = self.qbn(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)
        return x

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        
        self.qrelu.freeze(self.qi)
        self.qconv1.freeze(self.qi)
        self.qconv2.freeze(self.qconv1.qo)
        self.qbn.freeze(self.qconv2.qo)

    def quantize_inference(self, x):
        qx = self.qrelu.quantize_inference(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qbn.quantize_inference(qx)
        return qx
    
    def fake_quantize_inference(self, x):
        qx = self.qrelu.fake_quantize_inference(x)
        qx = self.qconv1.fake_quantize_inference(qx)
        qx = self.qconv2.fake_quantize_inference(qx)
        qx = self.qbn.fake_quantize_inference(qx)
        return qx

    
#added, QSepConv, 复合算子4
#如何传参？
class QSepConv(QModule):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, qi=False, qo=True, num_bits=8):
        super(QSepConv, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.qdconv1 = QDilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, \
                                 affine=affine, qi=False, qo=True, num_bits=8)
        self.qdconv2 = QDilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, \
                                 affine=affine, qi=False, qo=True, num_bits=8)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = self.qdconv1(x)
        x = self.qdconv2(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)
        return x

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        
        self.qdconv1.freeze(self.qi)
        self.qdconv2.freeze(self.qdconv1.qo)

    def quantize_inference(self, x):
        qx = self.qdconv1.quantize_inference(x)
        qx = self.qdconv2.quantize_inference(qx)
        return qx
    
    def fake_quantize_inference(self, x):
        qx = self.qdconv1.fake_quantize_inference(x)
        qx = self.qdconv2.fake_quantize_inference(qx)
        return qx
    

# 特殊量化算子，包括concate, eltwiseadd, softmax(废弃)
# 1.added, QConcate算子, 特殊算子1;
# 2.注意,concate不是nn.Module算子，
# 3.需要改成多操作数形式;
# 4.这里M不使用buffer会不会出问题, 可能造成RuntimeError:微分问题
# 4.未验证！
class QConcate(QModule2):
    def __init__(self, dim=1, qi_list=False, qo=True, num_bits=8):   # 此处qi和qo是Bool类
        super(QConcate, self).__init__(qi_list=qi_list, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.dim = dim

    def freeze(self, qi_list=None, qo=None):  # 此处qi和qo是QParam类
        if hasattr(self, 'qi_list') and qi_list is not None:
            raise ValueError('qi_list has been provided in init function.')

        if not hasattr(self, 'qi_list') and qi_list is None:
            raise ValueError('qi_list is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        #此处只考虑init中的qi_list为假，并且freeze中的qi_list非空
        #未考虑前者为真，且后者非空的情况，此情况会出Bug
        if qi_list is not None:
            self.qi_list = []
            for qi in qi_list:
                self.qi_list.append(qi)

        if qo is not None:
            self.qo = qo

        self.M_list = []
        for qi in self.qi_list:
            self.M_list.append((qi.scale / self.qo.scale).data)
        

    def forward(self, x_list):  # !在量化训练时候用
        if hasattr(self, 'qi_list'):
            for i in range(len(x_list)):
                self.qi_list[i].update(x_list[i])
                x_list[i] = FakeQuantize.apply(x_list[i], self.qi_list[i])
        
        x = torch.cat(x_list, self.dim)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x_list):  # !在推理时候用
        #两操作数改成多操作数
        # x1 = x1 - self.q1.zero_point
        # x2 = x2 - self.q2.zero_point
        # x1 = self.M1 * x1
        # x2 = self.M2 * x2
        # x1 = x1 + self.qo.zero_point
        # x2 = x2 + self.qo.zero_point
        for i in range(len(x_list)):
            x_list[i] = x_list[i] - self.qi_list[i].zero_point
            x_list[i] = self.M_list[i] * x_list[i]
            x_list[i] = x_list[i] + self.qo.zero_point

        x = torch.cat(x_list, self.dim)
        x.round_()
        x.clamp_(0., 2. ** self.num_bits - 1.).round_()
        return x
    
    def fake_quantize_inference(self, x_list):  # !在量化训练时候用
        if hasattr(self, 'qi_list'):
            for i in range(len(x_list)):
                x_list[i] = FakeQuantize.apply(x_list[i], self.qi_list[i])
        
        x = torch.cat(x_list, self.dim)

        if hasattr(self, 'qo'):
            x = FakeQuantize.apply(x, self.qo)
        return x

#added, QEltwiseAdd, 特殊算子2, 输入为任意元素个数的列表(每元素的形状相同)
class QEltwiseAdd(QModule2):
    def __init__(self, qi_list=False, qo=True, num_bits=8):   # 此处qi和qo是Bool类
        super(QEltwiseAdd, self).__init__(qi_list=qi_list, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        

    def freeze(self, qi_list=None, qo=None):  # 此处qi和qo是QParam类
        if hasattr(self, 'qi_list') and qi_list is not None:
            #test
            print(self,'\n' , self.qi_list, self.qi_flag)
            raise ValueError('qi_list has been provided in init function.')

        if not hasattr(self, 'qi_list') and qi_list is None:
            raise ValueError('qi_list is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi_list is not None:
            self.qi_list = []
            for qi in qi_list:
                self.qi_list.append(qi)

        if qo is not None:
            self.qo = qo

        self.M_list = []
        for i in range(len(self.qi_list)):
            self.M_list.append((self.qi_list[i].scale / self.qi_list[0].scale).data)
        #特殊，这个M是服务于进行加法运算后的数
        self.M_list.append((qi_list[0].scale / self.qo.scale).data)

    def forward(self, x_list):  # !在量化训练时候用
        if hasattr(self, 'qi_list'):
            #bug, qi_list[i]数组越界
            #test
            # print(len(self.qi_list), self.qi_list)
            for i in range(len(x_list)):
                self.qi_list[i].update(x_list[i])
                x_list[i] = FakeQuantize.apply(x_list[i], self.qi_list[i])
        
        #test, 问题出现在第二块cell中的s0和s1的尺寸不匹配
        # for i in range(len(x_list)):
        #     print(x_list[i].shape)
        # print(len(x_list), x_list[-1].shape)
        # if(len(x_list)==2):
        #     for i in range(len(x_list)):
        #         print(x_list[i].shape)
        x = sum(x_list)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x_list):  # !在推理时候用

        for i in range(len(x_list)):
            x_list[i] = x_list[i] - self.qi_list[i].zero_point
            x_list[i] = self.M_list[i] * x_list[i]
            
        x = sum(x_list)
        x = self.M_list[-1] * x

        x = x + self.qo.zero_point
        x.round_()
        x.clamp_(0., 2. ** self.num_bits - 1.).round_()
        return x
    
    def fake_quantize_inference(self, x_list):  # !在量化训练时候用
        if hasattr(self, 'qi_list'):
            for i in range(len(x_list)):
                x_list[i] = FakeQuantize.apply(x_list[i], self.qi_list[i])
        
        x = sum(x_list)

        if hasattr(self, 'qo'):
            x = FakeQuantize.apply(x, self.qo)
        return x

# added, QSoftmax, 特殊算子3,也算是复合算子中的一种, 输入为tensor, 这里暂时废弃
# 打算：对exp写一个量化算子QExp，再写一个div量化算子QDiv。
#  首先对x进行exp量化后，然后对分母进行sum量化，然后对分子分母进行div量化
# 注意：div量化将引入新的量化误差

#added, QExp
#如果要计算QSoftmax，则QExp输入是x_hat,即x-x_max
class QExp(QModule):
    def __init__(self, qi=False, qo=True, num_bits=8):
        super(QConv2d, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.register_buffer('M', torch.tensor([], requires_grad=False))
        self.coef = [0.3585, 1.353, 0.344]

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.q_b = torch.floor(self.coef[1] / self.qi.scale)
        self.q_c = torch.floor(self.coef[2] / self.qi.scale ** 2 /self.coef[0])

        self.q_ln2 = torch.log(torch.tensor([2]))/self.qi.scale+self.qi.zero_point
        self.q_ln2 = self.q_ln2.clamp_(0., 2. ** self.num_bits - 1.).round_()

        self.M.data = (self.coef[0] * self.qi.scale**2 / self.qo.scale).data


    def forward(self, x): 
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = torch.exp(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x):

        #因为本算子是单纯的exp(x)而不是exp(x-x.max)
        # x_max, _ = torch.max(x, dim=-1)
        # x_hat = x - x_max

        #因为推理时不需要求导，所以此处floor()函数没有进行ste化
        z = torch.floor(-(x-self.qi.zero_point)/(self.q_ln2-self.qi.zero_point))
        p_int = x + (self.q_ln2-self.qi.zero_point)*z
        p_int = p_int.clamp_(0., 2. ** self.num_bits - 1.).round_()

        x_poly_int = (p_int-self.qi.zero_point+self.q_b)**2+self.q_c
        x = x_poly_int * 2**(-z) * self.M
        
        x.round_()
        x = x + self.qo.zero_point
        x.clamp_(0., 2. ** self.num_bits - 1.).round_()
        return x

#added, QDiv, 为了计算QSoftmax
class QDiv(QModule2):
    def __init__(self, qi=False, qo=True, num_bits=8):   
        super(QDiv, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.register_buffer('M', torch.tensor([], requires_grad=False))

    def freeze(self, qi_list=None, qo=None):
        if hasattr(self, 'qi_list') and qi_list is not None:
            raise ValueError('qi_list has been provided in init function.')

        if not hasattr(self, 'qi_list') and qi_list is None:
            raise ValueError('qi_list is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi_list is not None:
            for i, qi in enumerate(qi_list):
                self.qi_list[i] = qi

        if qo is not None:
            self.qo = qo

        for qi in qi_list:
            self.M.data = (self.qi_list[0].scale / self.qi_list[1].scale / self.qo.scale).data

    def forward(self, x_list):  # !在量化训练时候用
        if hasattr(self, 'qi_list'):
            for i, xi in enumerate(x_list):
                self.qi_list[i].update(xi)
                x_list[i] = FakeQuantize.apply(xi, self.qi_list[i])
        
        x = torch.div(x_list[0], x_list[1])

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x_list):  
        x_list[0] = x_list[0] - self.qi_list[0].zero_point
        x_list[1] = x_list[1] - self.qi_list[1].zero_point

        x = torch.div(x_list[0], x_list[1]) 
        x = self.M * x
        x.round_()
        x = x + self.qo.zero_point
        x.clamp_(0., 2. ** self.num_bits - 1.).round_()
        return x
    
    def fake_quantize_inference(self, x_list):  # !在量化训练时候用
        if hasattr(self, 'qi_list'):
            for i, xi in enumerate(x_list):
                x_list[i] = FakeQuantize.apply(xi, self.qi_list[i])
        
        x = torch.div(x_list[0], x_list[1])

        if hasattr(self, 'qo'):
            x = FakeQuantize.apply(x, self.qo)

        return x

#added, QSoftmax, 复合算子。 其为exp量化算子 + eletwiseAdd量化算子 + div量化算子的特殊算子
class QSoftmax(QModule):
    def __init__(self, qi=False, qo=True, num_bits=8):
        super(QSoftmax, self).__init__(qi=qi, qo=qo, num_bits=num_bits)

        self.num_bits = num_bits
        self.qexp = QExp(qi=False, num_bits=8)
        self.qadd = QEltwiseAdd(qi=False,\
         qo=True, num_bits=8)
        self.qdiv = QDiv(qi_list=False, qo=True, num_bits=8)

    def forward(self, x):
        x1 = self.qexp(x)
        x2 = self.qadd(x1.tolist())
        x = self.qdiv([x1, x2])

        return x

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.qexp.freeze(self.qi)
        self.qadd.freeze([self.qexp.qo.clone() for _ in range(8)])
        self.qdiv.freeze([self.qexp.qo, self.qadd.qo])

        self.qo = self.qconcat.qo

    def quantize_inference(self, x):
        qx1 = self.qexp.quantize_inference(x)
        qx2 = self.qadd.quantize_inference(qx1.tolist())
        qx = self.qdiv.quantize_inference([qx1, qx2])
        return qx

#复合算子5
#added, QFactorizedReduce, 复合算子，不涉及量化参数的计算，只涉及传参
class QFactorizedReduce(QModule):
    def __init__(self, C_in, C_out, affine=True, qi=False, qo=True, num_bits=8):
        super(QFactorizedReduce, self).__init__(qi=qi, qo=qo, num_bits=num_bits)

        conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        bn_module = BatchNorm2d(C_out, affine=affine)

        self.num_bits = num_bits
        self.qrelu = QReLU(qi=False, num_bits=8)
        self.qconv1 = QConv2d(conv1, qi=False, qo=True, num_bits=8)
        self.qconv2 = QConv2d(conv2, qi=False, qo=True, num_bits=8)
        # self.qconcat = QConcate(dim=1, qi_list=[True, True], qo=True, num_bits=8)
        self.qconcat = QConcate(dim=1, qi_list=False, qo=True, num_bits=8)
        self.qbn = QBatchNorm2d(bn_module, qi=False, qo=True, num_bits=8)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = self.qrelu(x)
        x1 = self.qconv1(x)
        x2 = self.qconv2(x[:, :, 1:, 1:])#注意
        x = self.qconcat([x1, x2])
        x = self.qbn(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)
        return x

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.qrelu.freeze(self.qi)
        self.qconv1.freeze(self.qi)
        self.qconv2.freeze(self.qi)
        self.qconcat.freeze([self.qconv1.qo, self.qconv2.qo])
        self.qbn.freeze(self.qconcat.qo)

    def quantize_inference(self, x):
        qx = self.qrelu.quantize_inference(x)
        qx1 = self.qconv1.quantize_inference(qx)
        qx2 = self.qconv2.quantize_inference(qx[:, :, 1:, 1:])
        qx = self.qconcat.quantize_inference([qx1, qx2])
        qx = self.qbn.quantize_inference(qx)
        return qx
    
    def fake_quantize_inference(self, x):
        qx = self.qrelu.fake_quantize_inference(x)
        qx1 = self.qconv1.fake_quantize_inference(qx)
        qx2 = self.qconv2.fake_quantize_inference(qx[:, :, 1:, 1:])
        qx = self.qconcat.fake_quantize_inference([qx1, qx2])
        qx = self.qbn.fake_quantize_inference(qx)
        return qx
    
#added, QMixedOp, 特殊算子,候选量化算子+类似QEltwiseAdd，修正1次
#第1次：对于输入x对应qi，对于中间输入x_list对应qi_list
#因为mixedOp的输入x是一样的，所以qi_list的每一个元素也应该一样
class QMixedOp(QModule):
    def __init__(self, C, stride, qi=False, qo=True, num_bits=8):
        super(QMixedOp, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.qw = QParam(num_bits=num_bits)
        #bug？不是。 可能self.w如果不在初始化声明，那在传值给optim的时候就排除了它，导致计算梯度时的链式法则从本算子往输入节点方向都是错的，
        # 从而导致don't fit the gd code? 不对，在QSearchCell中self.qDag就是QMixedOp的w
        self.w = F.softmax(torch.randn(8), dim=-1)
        # test
        # print('\n\n++++\n', self.w, '\n', type(self.w), '\n', self.w.shape, '\n')

        # self.qi_list = [QParam(num_bits=num_bits) for i in range(8)]

        #第1次更新：
        self._qOps = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            qOp = QOPS[primitive](C, stride, affine=False)
            self._qOps.append(qOp)

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        
        self.qi_list = [copy.deepcopy(qi) for _ in range(8)]

        if qo is not None:
            self.qo = qo

        self.M_list = []
        for i in range(len(self.qi_list)):
            #bug fixed, 为了qOp.quantize_inference时QBatchNorm推理时qi不存在
            self._qOps[i].freeze(qi=self.qi)
            self.M_list.append((self.qi_list[i].scale / self.qi_list[0].scale).data)
        #特殊，这个M是服务于进行加法运算后的数
        self.M_list.append((self.qi_list[0].scale * self.qw.scale / self.qo.scale).data)
        
        # for Wq
        self.w.data = self.qw.quantize_tensor(self.w.data)
        self.w.data = self.w.data - self.qw.zero_point


    def forward(self, x, w):  # 一共三步
        # test, pass
        # print('------------------------\nQMixedOp foward!')
        #1.量化反量化x
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        #2.获取8个候选算子的输出x_list
        x_list = []
        for op in self._qOps:
            x_list.append(op(x))
            #test bug
            # print(op)
            
        
        if hasattr(self, 'qi_list'):
            for i in range(len(x_list)):
                self.qi_list[i].update(x_list[i])
                x_list[i] = FakeQuantize.apply(x_list[i], self.qi_list[i])

        #这里我们对softmax后的alpha即w的处理方式和普通w的一样，而不使用以下的处理
        # for qw
        # bug prob, w实际量纲为[0,1]，qw应该是个固定数值，这里我没使用固定数值，可能出错。（废弃）
        #self.qw.update(w)
        #self.qw.update(torch.tensor[0., 1.])

        #3.执行sum(w*xi)
        self.w = w
        # test
        # print('\n\n+++++++\n', type(w), '\n', w.shape, '\n', w)

        #bug fixed, 只有叶子节点能设置可不可导
        # self.w.requires_grad = True
        self.qw.update(w.data)
        FakeQuantize.apply(w, self.qw)
        
        #此处无法看出到底对谁进行backward，可能导致以下错误：
        #RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. 
        x = sum(w[i]*x_list[i] for i in range(len(x_list)))

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x):
        #1.经过8个候选算子，获取的输出为x_list
        x_list = []
        for qOp in self._qOps:
            x_list.append(qOp.quantize_inference(x))

        for i in range(len(x_list)):
            x_list[i] = x_list[i] - self.qi_list[i].zero_point
            x_list[i] = self.M_list[i] * x_list[i] * self.w[i]
            
        x = sum(x_list)
        x = self.M_list[-1] * x

        x = x + self.qo.zero_point
        x.round_()
        x.clamp_(0., 2. ** self.num_bits - 1.).round_()
        return x
    
    def fake_quantize_inference(self, x, w):  # 一共三步
        # test
        # print('------------------------\nQMixedOp fake_quantize_inference!')
        #1.量化反量化x
        if hasattr(self, 'qi'):
            x = FakeQuantize.apply(x, self.qi)

        #2.获取8个候选算子的输出x_list
        x_list = []
        for op in self._qOps:
            x_list.append(op(x))
               
        if hasattr(self, 'qi_list'):
            for i in range(len(x_list)):
                x_list[i] = FakeQuantize.apply(x_list[i], self.qi_list[i])

        #3.执行sum(w*xi)
        FakeQuantize.apply(w, self.qw)
        
        x = sum(w[i]*x_list[i] for i in range(len(x_list)))

        if hasattr(self, 'qo'):
            x = FakeQuantize.apply(x, self.qo)

        return x