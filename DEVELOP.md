# 开发日志

1. next: 确认需要重写哪些算子为量化算子?（看主函数训练和离散化时使用了哪些算子，主要看darts代码ops部分）
    - 看genotypes.py中的parse
    > 为什么将连续网络变成离散网络时，不将none候选算子考虑进去？

2. next: 量化算子具体怎么写（看白皮书和torch量化源码算子融合部分）
    - 多算子（看量化源码module.py中QConvBNReLU部分）
    > 包括PoolBN✔、StdConv✔、FacConv✔、DilConv✔、SepConv✔、FactorizedReduce✔、MixedOp✔
    >>PoolBN的量化算子怎么写？
    >>>本质上，它就是如何量化bn层算子的问题。它区别于矩阵运算的量化。在这里我们只关注其输入和输出，至于bn中的细节我们不管;而pooling操作不包含矩阵运算（可以看做特殊的矩阵运算），所以没有qw参数吗？（有qw,是gamma的量化参数）但是bn层包含per-channel操作，这会不会导致量化的时候必须也要per-channel？（是的，在量化的bn层是这样的。我只需在基本完成后，替换成cross-layer-equal）如何处理bn层的bias？
    <br/>关于per-channel量化：[per-channel原理--jermmy大佬](https://zhuanlan.zhihu.com/p/381679400)
    <br/>现在尝试对bn层量化，如果在demo中有效，那就在darts中使用
    >>
    >>>原理公式为：$y=\gamma\widetilde{x_i}+\beta$
    ($\widetilde{x_i}=\frac{x-\mu_B}{\sqrt{\sigma^2_B+\epsilon}}$)
    <br/>量化公式为：$s(q_3-z_3)=s_1(q_1-z_1)\frac{s_2(q_2-z_2)-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$
    >>
    >>>对于bn层的参数更新、计算量化参数，如何做？以及训练和推理怎么写？
    <br/>因为F.batch_norm操作不可避免的包含浮点运算，所以我要重写batch_norm，使得在推理时使用伪定点运算；又因为在感知量化训练时，对$\frac{s_1\times s_2}{s_3}$进行量化反量化后除以std的量化损失，和对$\frac{s_1\times s_2}{s_3\times std}$直接进行量化反量化的量化损失是不一样的(因为量化反量化没有scaling equivariance)
    <br/>因为bn层量化涉及到per-channel，公式如下：
    <br/>$q_3^{oc}=\sum\nolimits_{ic}\frac{s_1^{ic} s_2^{ic}}{s_3}\sum\nolimits_m\sum\nolimits_n(q_1^{ic}-z_1^{ic})(q_2^{ic}-z_2^{ic})+z_3$
    <br/>所以这需要EltwiseAdd量化算子？不需要
    <br/>改版的bn层通过训练损失测试！
    <br/>能否分别统一s1, s2, s3呢？似乎对于bn层这种特殊层（其gamma是每个channel只含一个元素），可以统一，这样per-channel变成了per-layer了；而对于别的像conv之类的层（其weight是每个channel是一个张量），不可以统一【仅对于理想状态的per-channel，见[per-channel原理--jermmy大佬](https://zhuanlan.zhihu.com/p/381679400)的公式(5)】
    <br/>感知量化在训练阶段loss损失很小，但是在推理过程中精度只有10，正在排查问题
    <br/>成功解决！原因是推理时的x = self.bn_module(x), 使得weight和bias被浮点数覆盖，导致推理前的freeze操作白费
    >>
    >>>处理方法：**直接通过$\gamma$之类的算出weight和bias，对其进行类似conv量化的操作，方式为per-tensor量化**。（对bn部分）理由是，虽然本质上bn层是只有一个filter、filter每channel的size为1的算子，如果量化应该使用per-channel，但是，其特殊性在于filter每channel的size为1，所以能够将filter的所有channel的量化参数统一，最终在效果上等同于per-tensor的（实际上是工程上的per-channel，但因为只有一个filter，不需要在最后做一次requantize）。注意，pool部分是不需要真正再做一个量化的，因为pool并不改变tensor的元素表示数值范围，只需要沿用上一层的量化参数，在训练时直接不量化反量化，在推理时不需要使用上freeze后的量化参数进行量化
    >
    >>StdConv的量化算子怎么写？
    >>>本质上，StdConv是relu->conv->bn，所以只需要借鉴conv->bn->relu的量化算子。但需要注意的是，推理时的clamp函数起到的作用不一定是relu的效果
    >
    >>FacConv的量化算子怎么写？
    >>>对于relu->conv->conv->bn，如何确定量化参数？我分成了四步看
    >
    >>DilConv的量化算子怎么写？
    >>>和FacConv是同样的结构，只是参数不同。因为需要区分QFacConv和QDilConv，以至于方便处理，所以虽然结构相同，但需要另外写成两种算子
    >
    >>FactorizedReduce的量化算子怎么写？
    >>>同FacConv量化算子，分成多步。看作复合算子，只负责传参不负责计算参数。(注意，conv2d中的输入是x[:, :, 1:, 1:])
    >
    >>MixedOp的量化算子怎么写？
    >>>本质上QMixedOp是**含n个元素的张量w每元素分别和n个张量做数乘**，可以理解为特殊的**逐元素乘法后相加**。
    >>
    >>>原理公式为$q_3=Z_3+\frac{S_w S_1}{S_3}(q_{w1}-Z_w)(q_1-Z_1)+\frac{S_w S_2}{S_3}(q_{w2}-Z_w)(q_2-Z_2)$
    <br/>最终公式为$q_3=Z_3+\frac{S_w S_1}{S_3}[(q_{w1}-Z_w)(q_1-Z_1)+\frac{S_2 }{S_1}(q_{w2}-Z_w)(q_2-Z_2)]$
    <br/>理由是w(即softmax之后的张量)是一个量纲，从而每元素共享同一个量化参数，所以在加法运算中对每个操作数分别乘上w的元素并不会改变操作数之间的相对量纲
    >>
    >>>实现：首先，造一个QOPS字典，然后写加法算子add,然后再写MixedOp
    >>
    >>>修复了QBatchNorm2d和QPoolBN在推理时没有使用推理状态bn参数的错误
    <br/>修改了QIdentity冗余代码
    <br/>修改了QFactorizedReduce的传参
    <br/>修改了QFacConv的传参
    <br/>修改了QDilConv的传参
    <br/>修改了QSepConv的传参
    <br/>既然量化推理时还需要输入w，那还有必要在训练时量化反量化吗？有必要，因为这是在模拟量化损失
    <br/>量化推理过程中的w到底是原数还是量化后的数？是量化后的数！关键在于qw存储了量化参数，w的量纲不变，所以w的量化参数也不变。 bug prob，注意，这里我使用了tensor[0., 1.]量化qw参数，即qw是定值

    - 特殊算子
    > 包括add✔、concate✔，详情见[add和concate量化原理--jermmy大佬](https://zhuanlan.zhihu.com/p/336682366)；此外还有一个**softmax**✔
    >>concate的量化算子怎么写？
    >>>量化公式：$q_3=concat[\frac{S_1}{S_3}(q_1-Z_1)+Z_3,\frac{S_2}{S_3}(q_2-Z_2)+Z_3]$。注意量化误差存在于rescale和requant中,现在暂时不考虑量化误差的问题，等到出现了再解决。未验证
    >>
    >>>注意，需要写成**多操作数**的形式！已改为多操作数形式，但没验证过
    >
    >>add量化算子怎么写？
    >>>同上concate理，一样是多操作数
    >
    >>softmax量化算子怎么写？
    >>>网上说8bit中，ptq支持很差，而qat支持很好，为什么？怎么办？
    <br/>**div的量化算子怎么写**？
    <br/>直接写，然后测试有没有问题。
    >>
    >>>***打算对exp写一个量化算子，再写一个div量化算子。首先对x进行exp量化后，然后对分母进行sum量化，然后对分子分母进行div量化***
    >>
    >>>以下是关于exp量化算子怎么写，原理详情见[softmax量化原理--jermmy大佬](https://zhuanlan.zhihu.com/p/587372438)，需要注意的问题见[关于softmax函数本身的问题和解决方法](https://cloud.tencent.com/developer/article/1157132)。原理公式如下：
    <br/>量化前：$softmax(x)_i=\frac{exp(x_i-x_{max})}{\sum_{j=1}^k exp(x_j-x_{max})}$, 令$\hat{x_i}=x_i-x_{max}$
    <br/>其中有：$exp(\hat{x}_i)=S_{out}(q_{out}-Z_{out})=2^{-z}aS_{in}^2[(q_{p}+\frac{b}{S_{in}})^2+\frac{c}{aS_{in}^2}]$，从而$q_{out}=\frac{2^{-z}aS_{in}^2}{S_{out}}[((q_{in}+q_{ln2}*z-Z_{in})+\frac{b}{S_{in}})^2+\frac{c}{aS_{in}^2}]+Z_{out}$
    <br/>量化后：$softmax(\hat{x_i})=S_{out}(q_{out}-Z_{out})=\frac{S_{exp}(q_{exp}-Z_{exp})}{S_{sum}(q_{sum}-Z_{sum})}$，从而$q_{out}=\frac{S_{exp}(q_{exp}-Z_{exp})}{S_{sum}S_{out}(q_{sum}-Z_{sum})}+Z_{out}$
    >>
    >>>模拟现象：8bit下，10000次近似的平均量化误差是10的-3次方级；32bit下，是10的-4次方级
    <br/>1.此处有个很坑的地方，我需要计算z，z是除法中的商，这就意味着量化时候必须严格遵循正负关系，即负数x_hat映射到range负数部分等比例的地方，正数x_hat映射到range正数部分等比例地方
    <br/>2.另一个坑是， x_poly_int即量化推理时使用的poly近似的部分输出数值很大，超出range范围,需要一个trick，就是在clamp之前，右移，问题在于移动多少位。最终我直接在计算完所有和poly相关部分并且执行2**(-z)之后再clamp。
    <br/>3.问题，exp(x_hat)的值域是(0,1]，而我8bit下计算出来它们的误差是10的-3级，这样的量化误差对于(0,1]可是不小
    >>
    >>>补充：div量化公式：$S_{out}(q_{out}-Z_{out}) = \frac{S_1(q_1-Z_1)}{S_2(q_2-Z_2)}$,从而$q_{out} = \frac{S_1(q_1-Z_1)}{S_2S_{out}(q_2-Z_2)}+Z_{out}$

    - 单算子(一轮)✔
    > 包括MaxPool2d✔、AvgPool2d✔、BatchNorm2d~~❌~~✔(二轮)、ReLU✔、Conv2d✔、Identity✔、Zero✔、AdaptiveAvgPool2d✔、Linear✔
    >>BatchNorm2d的量化算子怎么写？(目前暂时不对Bn层独立量化，因为没必要，也因为如果对bn层进行量化，那就必须使用per-channel量化，bn层的训练参数是对应不同channel有不同的gamma和beta，bn层的均值和方差也是如此)
    >>>怎么确定scale和zero point？（废弃）
    >>恢复BatchNorm2d量化算子
    >>>因为mixop中的一部分是PoolBN，所以不能舍弃batchnorm的量化算子。处理方法见多算子中的PoolBN算子
    >
    >>AvgPool2d的量化算子怎么写？
    >>>怎么确定scale和zero point?
    同MaxPool2d
    >
    >>AdaptiveAvgPool2d的量化算子怎么写？
    >>>如何确定输入参数output_size?这区别于AvgPool2d的kernel_size、stride和padding参数,同AvgPool2d
    >
    >>identity的量化算子怎么写?
    >>>搞清楚量化参数qi,qw,qo在identity中的必要性：forward()时，qi,qw分别量化反量化x、w, 然后op(w,x), 然后qo量化反量化y，显然qi、qw、qo不必要，但为了保持格式的一致性，要保留qi、qw、qo，只要在forward()时，qi，qw分别量化反量化x、w，然后空操作，然后令qo=qi（这里不再次进行量化反量化是为了减少可能产生的新的量化误差，但真的会出现新的量化误差吗？）-> 最后我是， 使用qi量化反量化x，然后空操作，然后使用qo量化反量化
    >
    >>Zero的量化算子怎么写？（注意别的环境会有1/0的bug，只是在python中不是bug, 1/0.=inf）
    >>>如何写F.Zero的函数，并将其使用在forward()中？<br/>！！！最最最坑爹的是，经过zero算子之后，输出全是0，这意味着max-min必为0. 所以，qo.scale=inf。然而，1/inf得到0. 这在python中不同于数学，它自行处理了。-> 我暂时不对zero做改变，对于python的运行环境来说，这不算一个bug（当然，如果换成其他的环境，那么很有可能是一个Bug）

3. next:整合量化代码
    - 2023.9.18完成一轮传参✔
4. next: debug
>关于NaN出现的问题：
>>不同的epoch应该使用独立的QParam吗？为什么？
>
>>为什么上一次epoch的freeze函数会对下一epoch的权重参数产生影响
>>>因为改变了weight和Bias
>
>>**已修复freeze带来的bug**

5. 关于LSQ的实现
>原理
>>[LSQ简单解释](https://zhuanlan.zhihu.com/p/406891271)
<br/>[**LSQ详细原理-jermmy大佬**](https://zhuanlan.zhihu.com/p/396001177)