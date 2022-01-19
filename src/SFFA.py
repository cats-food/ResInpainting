import torch
import torch.nn as nn
import torch.nn.functional as F


class SFFA(nn.Module):
    def __init__(self, patch_size=3, propagate_size=3, stride=1):
        super(SFFA, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None


    def forward(self, inputs, mask):
        bz, nc, h, w = inputs.size()  # batchsize, 通道数， 高，宽
        background = inputs * mask
        conv_kernels_all = background.view(bz, nc, w * h, 1, 1)  # view和reshape一样其实
        conv_kernels_all = conv_kernels_all.permute(0, 2, 1, 3, 4)  # 改变维度顺序，成了 bz * wh * nc * 1 * 1 的tensor
        output_tensor = []
        for i in range(bz):
            feature_map = inputs[i:i + 1]  # 被卷积的特征图，尺寸为 1*nc*w*h。索引[i:i+1]的作用是，设a是4维张量，a[i:i+1]得到的仍然是4维（保留维度），而a[i]得到的是3维（降一维），它们的数值是一样的。


            conv_kernels = conv_kernels_all[i] + 0.0000001  # wh*nc*1*1  加很小的数是防止后面求sqrt在0处不可导
            norm_factor = torch.sqrt(torch.sum(conv_kernels ** 2, [1, 2, 3], keepdim=True))  # wh*1*1*1
            conv_kernels_norm = conv_kernels / norm_factor  # wh*nc*1*1，可理解为wh个 nc*1*1的卷积核，相当于每个像素都被抽出来当一个卷积核

            conv_result = F.conv2d(feature_map, conv_kernels_norm, padding=self.patch_size // 2)

            # one-hot
            #attention_scores.scatter_(1, conv_result.argmax(dim=1, keepdim=True), 1)

            # softmax
            attention_scores = F.softmax(conv_result, dim=1)  # 在通道维度上做softmax，得到文中的socre'，尺寸为 1*wh*w*h
            feature_map = background[i:i + 1] + F.conv_transpose2d(attention_scores, conv_kernels, stride=1, padding=self.patch_size // 2)*(1-mask[i:i+1])  # src region remains
            output_tensor.append(feature_map)

        return torch.cat(output_tensor, dim=0)  # bz*nc*w*h 和輸入一樣


class SFFA_Module(nn.Module):

    def __init__(self, inchannel, patch_size_list=[1], propagate_size_list=[3], stride_list=[1]):
        assert isinstance(patch_size_list,
                          list), "patch_size should be a list containing scales, or you should use Contextual Attention to initialize your module"
        assert len(patch_size_list) == len(propagate_size_list) and len(propagate_size_list) == len(
            stride_list), "the input_lists should have same lengths"
        super(SFFA_Module, self).__init__()

        self.tgt_att = SFFA(patch_size_list[0], propagate_size_list[0], stride_list[0])
        self.num_of_modules = len(patch_size_list)
        self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)

    def forward(self, inputs, mask):
        outputs = self.tgt_att(inputs, mask)
        outputs = torch.cat([outputs, inputs], dim=1)
        outputs = self.combiner(outputs)
        return outputs

