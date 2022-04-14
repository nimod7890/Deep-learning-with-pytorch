import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias
    ):
        super(conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)
        return outputs


class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super(FeatureMap_convolution, self).__init__()
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = (
            3,
            64,
            3,
            2,
            1,
            1,
            False,
        )
        self.cbnr_1 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias
        )

        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = (
            64,
            64,
            3,
            1,
            1,
            1,
            False,
        )
        self.cbnr_2 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias
        )

        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = (
            64,
            128,
            3,
            1,
            1,
            1,
            False,
        )
        self.cbnr_3 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        outputs = self.maxpool(x)
        return outputs


class conv2DBatchNorm(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias
    ):
        super(conv2DBatchNorm, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)
        return outputs


class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(bottleNeckPSP, self).__init__()
        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, 1, 0, 1, False)
        self.cbr_2 = conv2DBatchNormRelu(
            mid_channels, mid_channels, 3, stride, dilation, dilation, False
        )
        self.cbr_3 = conv2DBatchNormRelu(mid_channels, out_channels, 1, 1, 0, 1, False)

        self.cb_residual = conv2DBatchNorm(
            in_channels, out_channels, 1, stride, 0, 1, False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbr_3(self.cbr_2(self.cbr_1(x)))
        residual = self.cb_residual(x)
        return self.relu(conv + residual)


class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, dilation):
        super(bottleNeckIdentifyPSP, self).__init__()
        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, 1, 0, 1, False)
        self.cbr_2 = conv2DBatchNormRelu(
            mid_channels, mid_channels, 3, 1, dilation, dilation, False
        )
        self.cbr_3 = conv2DBatchNorm(mid_channels, in_channels, 1, 1, 0, 1, False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbr_3(self.cbr_2(self.cbr_1(x)))
        # x=residual
        return self.relu(conv + x)


class ResidualBlockPSP(nn.Sequential):  # nn.Sequential 상속 시 forward가 이미 구현되어 있음
    def __init__(
        self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation
    ):
        super(ResidualBlockPSP, self).__init__()
        self.add_module(
            "block1",
            bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation),
        )
        for i in range(n_blocks - 1):
            self.add_module(
                "block" + str(i + 2),
                bottleNeckIdentifyPSP(out_channels, mid_channels, dilation),
            )


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, height, width):
        super(PyramidPooling, self).__init__()
        self.height = height
        self.width = width

        out_channels = int(in_channels / len(pool_sizes))
        self.avpool = []
        self.cbr = []
        for pool_size in pool_sizes:
            self.avpool.append(nn.AdaptiveAvgPool2d(output_size=pool_size))
            self.cbr.append(
                conv2DBatchNormRelu(in_channels, out_channels, 1, 1, 0, 1, False)
            )

    def forward(self, x):
        outList = [x]
        for pool, cbr in zip(self.avpool, self.cbr):
            out = cbr(pool(x))
            out = F.interpolate(
                out, size=(self.height, self.width), mode="bilinear", align_corners=True
            )
            outList.append(out)
        output = torch.cat(outList, dim=1)
        return output


class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_classes):
        super(DecodePSPFeature, self).__init__()
        self.height = height
        self.width = width

        self.cbr = conv2DBatchNormRelu(4096, 512, 3, 1, 1, 1, False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(512, n_classes, 1, 1, 0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(
            x, size=(self.height, self.width), mode="bilinear", align_corners=True
        )
        return output


class AuxiliaryPSPLayers(nn.Module):
    def __init__(self, in_channels, height, width, n_classes):
        super(AuxiliaryPSPLayers, self).__init__()
        self.height = height
        self.width = width
        self.cbr = conv2DBatchNormRelu(in_channels, 256, 3, 1, 1, 1, False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(256, n_classes, 1, 1, 0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(
            x, size=(self.height, self.width), mode="bilinear", align_corners=True
        )
        return output


class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super(PSPNet, self).__init__()
        block_config = [3, 4, 6, 3]
        img_size = 475
        img_size_8 = 60  # img_size/8

        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP(block_config[0], 128, 64, 256, 1, 1)
        self.feature_res_2 = ResidualBlockPSP(block_config[1], 256, 128, 512, 2, 1)
        self.feature_dilated_res_1 = ResidualBlockPSP(
            block_config[2], 512, 256, 1024, 1, 2
        )
        self.feature_dilated_res_2 = ResidualBlockPSP(
            block_config[3], 1024, 512, 2048, 1, 4
        )

        self.pyramid_pooling = PyramidPooling(
            2048, [6, 3, 2, 1], img_size_8, img_size_8
        )
        self.decode_feature = DecodePSPFeature(img_size, img_size, n_classes)
        self.aux = AuxiliaryPSPLayers(1024, img_size, img_size, n_classes)

    def forward(self, x):

        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)
        output_aux = self.aux(x)

        x = self.feature_dilated_res_2(x)
        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)

        return (output, output_aux)
