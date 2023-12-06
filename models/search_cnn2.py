""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell
from models.search_cells2 import QSearchCell
import genotypes as gt
from torch.nn.parallel._functions import Broadcast
import logging

from .module import *
import multiprocessing

def broadcast_list(l, device_ids):
    """ Broadcasting list """
    # print('---\n', l)
    l_copies = Broadcast.apply(device_ids, *l)
    # print('---\n', l_copies)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]
    # print('---\n', l_copies)

    return l_copies

#（2）重写model模块即search_cnn模块、search_cells以及ops；
class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        #de, 屏蔽非量化算子，weights包括量化算子的也包括非量化算子的，此处尝试不生成量化算子的
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        #added, it's ok
        self.n_nodes = n_nodes
        self.C_cur = C_cur
        #

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        #de,屏蔽非量化算子，weights包括量化算子的也包括非量化算子的，此处尝试不生成量化算子的
        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            # when cell is reduction type, C_cur multiply 2
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    #de, 屏蔽非量化算子，weights包括量化算子的也包括非量化算子的，此处尝试不生成量化算子的
    def forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits
    
    #added, CNN Quantization, start
    #如何处理qstem，到底需不需要拆开？inference时调用的qstem.qi是谁的qi？
    def quantize(self, num_bits=8):
        #1.获取qstem的量化算子
        conv1 = nn.Conv2d(self.C_in, self.C_cur, 3, 1, 1, bias=False)
        bn1 = BatchNorm2d(self.C_cur)
        # tmp test, qStem bug
        self.qStem = nn.Sequential(
            QConv2d(conv1, qi=True, qo=True, num_bits=num_bits),
            QBatchNorm2d(bn1, qi=False, qo=True, num_bits=num_bits)
        )
        # self.qStem = self.stem


        #2.获取所有的cell量化算子
        C_pp, C_p, C_cur = self.C_cur, self.C_cur, self.C

        self.qCells = nn.ModuleList()
        reduction_p = False
        for i in range(self.n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            # when cell is reduction type, C_cur multiply 2
            if i in [self.n_layers//3, 2*self.n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            # tmp test, bug qCell!
            qCell = QSearchCell(self.n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.qCells.append(qCell)
            # self.qCells = self.cells
            C_cur_out = C_cur * self.n_nodes
            C_pp, C_p = C_p, C_cur_out
        
        #3.收尾量化算子
        self.qGap =QAdaptiveAvgPool2d(1)
        self.qlinear = QLinear(self.linear, qi=False, qo=True, num_bits=num_bits)
        # self.qGap = self.gap
        # self.qlinear = self.linear

    def quantize_forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.qStem(x)

        for qCell in self.qCells:
            weights = weights_reduce if qCell.reduction else weights_normal
            s0, s1 = s1, qCell(s0, s1, weights)
            #test
            # print('s1 is ', s1.shape)

        out = self.qGap(s1)
        out = out.view(out.size(0), -1)
        logits = self.qlinear(out)
        return logits

    def freeze(self):
        #1.freeze stem
        self.qStem[0].freeze()
        self.qStem[1].freeze(self.qStem[0].qo)
        qi_s0 = qi_s1 = self.qStem[1].qo
        #2.freeze qCell
        for qCell in self.qCells:
            list_qi_cell = [qi_s0, qi_s1]
            #test
            # print(qCell.qi_list)
            qCell.freeze(list_qi_cell)
            qi_s0, qi_s1 = qi_s1, qCell.qo
        #3.freeze tail
        self.qGap.freeze(qi_s1)
        #test
        # print('self.qGap.qo is ', self.qGap.qo.scale, self.qGap.qo.zero_point)
        # bug fixed, qGap是pool量化算子，其不存在qo
        # self.qlinear.freeze(self.qGap.qo)
        self.qlinear.freeze(qi_s1)

    def cleanQi(self):
        
        #1.clean stem's qi
        self.qStem[0].cleanQi()
        self.qStem[1].cleanQi()

        #2.clean qCell's qi
        for qCell in self.qCells:
            qCell.cleanQi()

        #3.clean tail's qi
        self.qGap.cleanQi()
        self.qlinear.cleanQi()


    def quantize_inference(self, x):
        qx = self.qStem[0].qi.quantize_tensor(x)
        # s0 = s1 = self.qStem.quantize_inference(qx)#可能出错，没试过对nn.ModuleList执行量化推理
        qx = self.qStem[0].quantize_inference(qx)
        s0 = s1 = self.qStem[1].quantize_inference(qx)

        for qCell in self.qCells:
            s0, s1 = s1, qCell.quantize_inference(s0, s1)
        qx = self.qGap.quantize_inference(s1)
        qx = qx.view(qx.size(0), -1)
        qx = self.qlinear.quantize_inference(qx)
        out = self.qlinear.qo.dequantize_tensor(qx)

        return out
    #added, CNN Quantization, end

    def fake_quantize_inference(self, x, weights_normal, weights_reduce):
        # test
        # print('------------------------\ncnn fake quant inference!')
        qx = self.qStem[0].fake_quantize_inference(x)
        s0 = s1 = self.qStem[1].fake_quantize_inference(qx)

        for qCell in self.qCells:
            weights = weights_reduce if qCell.reduction else weights_normal
            s0, s1 = s1, qCell.fake_quantize_inference(s0, s1, weights)
        qx = self.qGap.fake_quantize_inference(s1)
        qx = qx.view(qx.size(0), -1)
        qx = self.qlinear.fake_quantize_inference(qx)

        return qx

class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    # self added, about params
    """
    C_in, input channels;
    C;
    n_classes, num of classes when predicting;
    n_layers, num of cells;
    criterion, class of loss function;
    n_nodes, num of intermediate nodes in a cell;
    stem_multiplier, togather with C, decides ouput channels of conv2d;
    """
    def __init__(self, C_in, C, n_classes, n_layers, criterion, n_nodes=4, stem_multiplier=3,
                 device_ids=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas, here the num is 8
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(n_nodes):
            # self.alpha_normal||alpha_reduce stores all node's alpha Tensors, please see log
            # Tensor format is row i+2, column n_ops
            self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))

        # setup alphas list
        """
        name_parameters:
        Returns an iterator over module parameters, 
        yielding both the name of the parameter as well as the parameter itself.
        """
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(C_in, C, n_classes, n_layers, n_nodes, stem_multiplier)

    def forward(self, x):

        # -1 means last dim, each row's elements(softmax(alpha)) add up to 1 
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        if len(self.device_ids) == 1:
            return self.net(x, weights_normal, weights_reduce)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies, wreduce_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

    #added, CNNController Quantization
    def quantize(self, num_bits=8):
        self.net.quantize(num_bits)

    def quantize_forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        return self.net.quantize_forward(x, weights_normal, weights_reduce)
    
        # wnormal_copies = Broadcast.apply(self.device_ids, *weights_normal)
        # wreduce_copies = Broadcast.apply(self.device_ids, *weights_reduce)
        # return self.net.quantize_forward(x, wnormal_copies, wreduce_copies)

        # 原版数据并行
        # if len(self.device_ids) == 1:
        #     return self.net.quantize_forward(x, weights_normal, weights_reduce)

        # scatter x
        # xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        # wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        # wreduce_copies = broadcast_list(weights_reduce, self.device_ids)
        # replicas = nn.parallel.replicate(self.net, self.device_ids)
        # outputs = nn.parallel.parallel_apply(list(replicas[i].quantize_forward for i in range(len(self.device_ids))),
        #                                      list(zip(xs, wnormal_copies, wreduce_copies)),
        #                                      devices=self.device_ids)
        # return nn.parallel.gather(outputs, self.device_ids[0])

    def freeze(self):
        self.net.freeze()
    
    #added
    def cleanQi(self):
        self.net.cleanQi()

    def quantize_inference(self, x):
        out = self.net.quantize_inference(x)
        return out
    
    def fake_quantize_inference(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        out = self.net.fake_quantize_inference(x, weights_normal, weights_reduce)
        return out
    #added, CNNController Quantization

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)
    
    #added, cal loss of qat
    def loss_quantized(self, X, y):
        logits = self.quantize_forward(X)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        # 每个结点的输入是两条确定边, gene_nomal's element is [(op1,i1),(op2,i2)]
        gene_normal = gt.parse(self.alpha_normal, k=2) 
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    #fixed,只返回量化算子的参数
    def weights(self):
        # return self.net.parameters()
        res = []
        for n, p in self.named_parameters():
            if 'stem' not in n and 'cell' not in n:
            # if 'cell' not in n:
                res.append(p)
        return res

    #fixed,只返回量化算子的参数
    def named_weights(self):
        # return self.net.named_parameters()
        res = []
        for n, p in self.named_parameters():
            if 'stem' not in n and 'cell' not in n:
            # if 'cell' not in n:
                res.append((n, p))
        return res

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
