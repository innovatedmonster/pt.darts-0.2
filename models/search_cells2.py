""" CNN cell for architecture search """
import torch
import torch.nn as nn
from models import ops
from .module import *
import copy

# QSearchCell是一个复合量化算子。
# 因为cell是cnn的部分，cell之间需要传递量化参数，所以cell必须是一个复合量化算子以传递量化参数
# 疑问，这里的affine为什么是false?
class QSearchCell(QModule2):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction, affine=False, qi_list=False, num_bits=8):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__(qi_list=qi_list, num_bits=num_bits)
        self.num_bits = num_bits
        self.reduction = reduction
        self.n_nodes = n_nodes
        
        if reduction_p:
            self.qPreproc0 = QFactorizedReduce(C_pp, C, affine=affine, \
                                               qi=False, qo=True, num_bits=num_bits)
        else:
            self.qPreproc0 = QStdConv(C_pp, C, 1, 1, 0, affine=affine, \
                                      qi=False, qo=True, num_bits=num_bits)
        self.qPreproc1 = QStdConv(C_p, C, 1, 1, 0, affine=affine, \
                                  qi=False, qo=True, num_bits=num_bits)

        #获取所有QMixedOp到aDag中，所有QEltwiseAdd到qSum中
        #qDag[i][j] 就是 第j个miexedOp
        self.qDag = nn.ModuleList()
        self.qSum = nn.ModuleList()
        for i in range(self.n_nodes):
            self.qDag.append(nn.ModuleList())
            self.qSum.append(QEltwiseAdd(qi_list=False, \
                                         qo=True, num_bits=num_bits))
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                qOp = QMixedOp(C, stride, qi=False, qo=True, num_bits=num_bits)
                self.qDag[i].append(qOp)
                # test fake
                # print(self.qDag[i][j].fake_quantize_inference, '\n', '\n')

        # 获取QConcate, cat所有的中间节点
        # bug fixed
        # self.qConcat = QConcate(dim=1, qi_list=[True for index in range(n_nodes)], \
        #                         qo=True, num_bits=num_bits)  
        self.qConcat = QConcate(dim=1, qi_list=False, \
                                qo=True, num_bits=num_bits)    

    def forward(self, s0, s1, w_dag):
        #self.weight = w_dag

        s0 = self.qPreproc0(s0)
        #bug fixed, s1的错误变形[32,16,32,17]->[32,16,32,2]
        s1 = self.qPreproc1(s1)

        qStates = [s0, s1]
        #test s0 s1
        # print('s0, s1 is: ',s0.shape, s1.shape)
        # qSates 是 所有前继节点的feature map
        # s_cur 是 当前节点的feature map
        # edges 是 QMixOps的列表
        # w_dag 是 所有QMixedIp的权重 
        for i, edges in enumerate(self.qDag):
            # s_cur = sum(edges[j](s, w) for j, (s, w) in enumerate(zip(qStates, w_list)))
            #test, edges[j](s, w)有bug
            # for j, s in enumerate(qStates):
            #     print(edges[j](s, w_dag[i][j]))
            #bug fixed, self.qSum[i]是QEltwiseAdd,输入应该是一个list，健壮性不如sum
            x_list = list(edges[j](s, w_dag[i][j]) for j, s in enumerate(qStates))
            s_cur = self.qSum[i](x_list)

            qStates.append(s_cur)

        # test intermediate
        # for i in range(len(qStates[2:])):
        #     print(qStates[2:][i].shape)
        # s_out = torch.cat(qStates[2:], dim=1)# concat all intermediate nodes
        s_out = self.qConcat(qStates[2:])

        # test s_out
        # print('s_out is: ', s_out.shape)
        return s_out

    def freeze(self, qi_list=None, qo=None):
        if hasattr(self, 'qi_list') and qi_list is not None:
            #test
            # print(self.qi_list)
            raise ValueError('qi_list has been provided in init function.')
        if not hasattr(self, 'qi_list') and qi_list is None:
            raise ValueError('qi_list is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi_list is not None:
            self.qi_list= []
            for i, qi in enumerate(qi_list):
                self.qi_list.append(qi)
        if qo is not None:
            self.qo = qo

        self.qPreproc0.freeze(qi=self.qi_list[0])
        #test
        # print('qi_list[0] is \n', self.qi_list[0], '\nqi_list[1] is \n', self.qi_list[1])
        self.qPreproc1.freeze(qi=self.qi_list[1])

        # 每个self.qDag[i][j]等价于一个mixedOp
        # qSum[i]之后得到的是一个qFeatureMap
        list_qi_qConcat = []
        for i, edges in enumerate(self.qDag):
            list_qi_qSum = []
            for j in range(len(edges)):# len(w_list)=前继节点的个数=len(edges)
                if j == 0:
                    self.qDag[i][j].freeze(qi=self.qPreproc0.qo)
                elif j == 1:
                    self.qDag[i][j].freeze(qi=self.qPreproc1.qo)
                else:
                    self.qDag[i][j].freeze(qi=self.qSum[j-2].qo)# 必须减去2，因为从两个输入开始才出现qSum
                list_qi_qSum.append(self.qDag[i][j].qo)

            #bug fixed, 是qCell调用cleanQi时没有调用qSum的cleanQi, 因为判断语句写成了仅QModule
            #test
            # print('qSum son number is ', len(list(self.qSum[i].children())))
            # if len(list(self.qSum[i].children())) != 0:
            #     for name, module in self.qSum[i].named_modules():
            #         if isinstance(module, QModule2):
            #             print('qSum[i] leaf is ', name, module)
            #         else:
            #             print('qSum[i] nonQ leaf is ', name, module)
            # print('qSum[i] qi_list is ', self.qSum[i].qi_list)#没有这个成员

            self.qSum[i].freeze(qi_list=list_qi_qSum)
            list_qi_qConcat.append(self.qSum[i].qo)
        
        self.qConcat.freeze(qi_list=list_qi_qConcat)
        self.qo = self.qConcat.qo

    # def cleanQi(self):
    #     if self.qi_flag is False and hasattr(self, 'qi_list'):
    #         del self.qi_list
    #     elif self.qi_flag is True and hasattr(self, 'qi_list'):
    #         for qi in self.qi_list:
    #             qi.min = torch.tensor([], requires_grad=False)
    #             qi.max = torch.tensor([], requires_grad=False)
    #     if hasattr(self, 'qw'):
    #         self.qw.min = torch.tensor([], requires_grad=False)
    #         self.qw.max = torch.tensor([], requires_grad=False)
    #     if hasattr(self, 'qo'):
    #         self.qo.min = torch.tensor([], requires_grad=False)
    #         self.qo.max = torch.tensor([], requires_grad=False)        

    #     self.qPreproc0.cleanQi()
    #     self.qPreproc1.cleanQi()

    #     for i, edges in enumerate(self.qDag):
    #         for j in range(len(edges)):# len(w_list)=前继节点的个数=len(edges)
    #             self.qDag[i][j].cleanQi()

    #         self.qSum[i].cleanQi()
        
    #     self.qConcat.cleanQi()

    def quantize_inference(self, s0, s1):
        qp1 = self.qPreproc0.quantize_inference(s0)
        qp2 = self.qPreproc1.quantize_inference(s1)
        
        qStates = [qp1, qp2]
        for i in range(len(self.qDag)):#确定二维的len会得到最外层的索引数
            list_qx_qSum = []
            for j, qx in enumerate(qStates):
                qOp_cur = self.qDag[i][j].quantize_inference(qx)
                list_qx_qSum.append(qOp_cur)
            s_cur = self.qSum[i].quantize_inference(list_qx_qSum)
            qStates.append(s_cur)
        
        out = self.qConcat.quantize_inference(qStates[2:])#cat所有中间节点
        return out
    
    def fake_quantize_inference(self, s0, s1, w_dag):
        # test, pass
        # print('------------------------\ncell fake quant inference!')
        qp1 = self.qPreproc0.fake_quantize_inference(s0)
        qp2 = self.qPreproc1.fake_quantize_inference(s1)
        
        qStates = [qp1, qp2]
        for i in range(len(self.qDag)):#确定二维的len会得到最外层的索引数
            list_qx_qSum = []
            for j, qx in enumerate(qStates):
                # test fake, bug, 没执行qDag[i][j]即混合算子的模拟量化推理,以及, 原因未知
                # print(self.qDag[i][j].fake_quantize_inference, '\n', '\n')
                qOp_cur = self.qDag[i][j].fake_quantize_inference(qx, w_dag[i][j])
                # qOp_cur = self.qDag[i][j].quantize_inference(qx)
                list_qx_qSum.append(qOp_cur)
            s_cur = self.qSum[i].fake_quantize_inference(list_qx_qSum)
            qStates.append(s_cur)
        
        out = self.qConcat.fake_quantize_inference(qStates[2:])#cat所有中间节点
        return out
    
        s0 = self.qPreproc0(s0)
        s1 = self.qPreproc1(s1)
        qStates = [s0, s1]
        for i, edges in enumerate(self.qDag):
            x_list = list(edges[j](s, w_dag[i][j]) for j, s in enumerate(qStates))
            s_cur = self.qSum[i](x_list)
            qStates.append(s_cur)
        s_out = self.qConcat(qStates[2:])

        return s_out