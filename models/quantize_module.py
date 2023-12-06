# 输入一个originalNAS的模型，输出一个继承其权重的量化模型
from search_quantize import *

def quantize_module(model_origin):
    # 生成量化模型
    _input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)
    net_crit = nn.CrossEntropyLoss()
    model_quantized = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus)
    # 继承originalNAS模型

    # 返回一个继承originalNAS权重的模型
    return model_quantized