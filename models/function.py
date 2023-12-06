from torch.autograd import Function


class FakeQuantize(Function):

    @staticmethod
    # for quantize_training's quantize and dequantize, when quantize_forward()
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    # for STE, straight through the non-continuous function's backward whose gradient is 1
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None