""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch

#注意，此处是量化版本的，计算loss过程(使用quantize_forward而不是forward)区别于原版
#疑问，Architect有什么用？
class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    #疑问：这里的virtual_step就是更新梯度，为啥不直接loss.backward()?
    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """

        # forward & calc loss
        #bug, loss_qat出现NaN
        #fix DataParallel
        loss_qat = self.net.module.loss_quantized(trn_X, trn_y) # L_trn(w)

        # 此处使用的self.net.weights()除了量化算子的，还包括非量化算子的，导致以下Bug
        # bug fixed: RuntimeError: One of the differentiated Tensors appears to 
        #   not have been used in the graph. 
        #   Set allow_unused=True if this is the desired behavior.
        # 只需要保证optim的输入参数是生成并参与计算图计算的，我将原版的非量化算子参数排除后解决了

        # test w_optim的输入参数
        # file_path = 'param_name.txt'
        # import os
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        # res = []
        # for n, p in self.net.named_parameters():
        #     if 'stem' not in n and 'cell' not in n:
        #         with open(file_path, 'a') as file:
        #             file.write(n + '\n')
        
        #bug NaN here!!!
        #test
        # 启用异常检测，没有作用，根本没检测出来，对backward和grad都不起作用
        # torch.autograd.detect_anomaly(True)
        # loss_qat.backward(retain_graph=True)

        # compute gradient: loss_qat对weights的导数
        #fix DataParallel
        gradients = torch.autograd.grad(loss_qat, self.net.module.weights())

        #test, gradients存在NaN
        # contains_nan = torch.isnan(gradients[0])
        # if contains_nan.any():
        #     print("张量包含 NaN 值")
        # else:
        #     print("张量不包含 NaN 值")

        #test
        # print('-------gradients:', gradients, '\n')
        # print('-------after grad:')
        # for alpha in self.net.alpha_normal:
        #     print(alpha.grad)

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            #fix DataParallel
            for w, vw, g in zip(self.net.module.weights(), self.v_net.module.weights(), gradients):
                #bug,当config.w_momentum被设置为0，get方法返回一个None,为什么？
                if w_optim.state[w].get('momentum_buffer', 0.) == None:
                    w_optim.state[w]['momentum_buffer'] = 0.

                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            #fix DataParallel
            for a, va in zip(self.net.module.alphas(), self.v_net.module.alphas()):
                va.copy_(a)

    #疑问：这里的unrolled是什么意思？
    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss_quantized
        #fix DataParallel
        loss_qat = self.v_net.module.loss_quantized(val_X, val_y) # L_val(w`)

        # compute gradient
        #fix DataParallel
        v_alphas = tuple(self.v_net.module.alphas())
        v_weights = tuple(self.v_net.module.weights())
        v_grads = torch.autograd.grad(loss_qat, v_alphas + v_weights)#为什么alpha + weights?
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():#这里为什么要禁止梯度计算（no_grad的作用是禁止梯度计算,但不会屏蔽梯度的数值）
            #fix DataParallel
            for alpha, da, h in zip(self.net.module.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h

    #注意，此处会改变self.net.weights()的数值，而不仅仅计算海森矩阵
    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            #fix DataParallel
            for p, d in zip(self.net.module.weights(), dw):
                p += eps * d
        #bug fixed, 这里没有使用量化版本的loss
        #fix DataParallel
        loss = self.net.module.loss_quantized(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.module.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            #fix DataParallel
            for p, d in zip(self.net.module.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.module.loss_quantized(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.module.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            #fix DataParallel
            for p, d in zip(self.net.module.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
