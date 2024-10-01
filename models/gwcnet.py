from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math

from models.attention import AttentionFuse
from models.refine import Refine

# 1
class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       Mish(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       Mish(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       Mish())

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        if not self.concat_feature:
            return [l2, l3, l4]
            # return {"gwc_feature": gwc_feature}
        else:
            gwc_feature = torch.cat((l2, l3, l4), dim=1)
            concat_feature = self.lastconv(gwc_feature)
            return [l2, l3, l4, concat_feature]
            # return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}

# 2
class attention_fuse(nn.Module):
    def __init__(self):
        super(attention_fuse, self).__init__()
        self.attn0 = AttentionFuse(64, 8, 3, True)
        self.attn1 = AttentionFuse(128, 16, 3, True)
        self.attn2 = AttentionFuse(128, 16, 3, True)

        self.attn3 = AttentionFuse(32, 4, 3, True)

    def forward(self, features_left, features_right):
        features_left[0], features_right[0] = self.attn0(features_left[0], features_right[0])
        features_left[1], features_right[1] = self.attn1(features_left[1], features_right[1])
        features_left[2], features_right[2] = self.attn2(features_left[2], features_right[2])

        features_left[3], features_right[3] = self.attn3(features_left[3], features_right[3])

        return features_left, features_right


class init_cost_volume(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(init_cost_volume, self).__init__()

        self.dres0 = nn.Sequential(convbn_3d(in_channels, mid_channels, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(mid_channels, mid_channels, 3, 1, 1),
                                   Mish())

        self.dres1 = nn.Sequential(convbn_3d(mid_channels, mid_channels, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(mid_channels, mid_channels, 3, 1, 1))

    def forward(self, volume):
        out = self.dres0(volume)
        out = self.dres1(out) + out
        return out


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   Mish())

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   Mish())

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        # conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        # conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        conv5 = FMish(self.conv5(conv4) + self.redir2(conv2))
        conv6 = FMish(self.conv6(conv5) + self.redir1(x))

        return conv6

class prediction(nn.Module):
    def __init__(self, in_channels):
        super(prediction, self).__init__()

        self.classif0 = nn.Sequential(convbn_3d(in_channels, in_channels, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(in_channels, 1, kernel_size=3, padding=1, stride=1, bias=False))

    def forward(self, out, maxdisp, h, w):
        cost_raw = self.classif0(out)
        cost = F.interpolate(cost_raw, [maxdisp, h, w], mode='trilinear')
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost, dim=1)
        pred = disparity_regression(pred, maxdisp)
        return pred, cost_raw

# 3
class cost_aggr_and_pred(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super(cost_aggr_and_pred, self).__init__()

        self.dres0 = init_cost_volume(in_channel, mid_channel)
        self.dres1 = hourglass(mid_channel)
        self.dres2 = hourglass(mid_channel)
        # self.dres3 = hourglass(mid_channel)

        self.prediction0 = prediction(mid_channel)
        self.prediction1 = prediction(mid_channel)
        self.prediction2 = prediction(mid_channel)
        # self.prediction3 = prediction(mid_channel)

    def forward(self, volume, maxdisp, h, w):
        out0 = self.dres0(volume)
        out1 = self.dres1(out0)
        out2 = self.dres2(out1)
        # out3 = self.dres3(out2)

        if self.training:
            pred0, cost0 = self.prediction0(out0, maxdisp, h, w)
            pred1, cost1 = self.prediction1(out1, maxdisp, h, w)
            pred2, cost2 = self.prediction2(out2, maxdisp, h, w)
            # pred3, cost3 = self.prediction3(out3, maxdisp, h, w)
            return [out0, out1, out2], [cost0, cost1, cost2], [pred0, pred1, pred2]

        else:
            # pred3, cost3 = self.prediction3(out3, maxdisp, h, w)
            return [out2], [], []


class GwcNet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume, use_attention_fuse):
        super(GwcNet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume
        self.use_attention_fuse = use_attention_fuse

        self.num_groups = 40

        self.refine = Refine()

        # gwcnet-g
        if not self.use_concat_volume and not self.use_attention_fuse:
            self.feature_extraction = feature_extraction(False)
            self.ca_and_pred_g = cost_aggr_and_pred(self.num_groups, 32)

        # gwcnet-a
        if not self.use_concat_volume and self.use_attention_fuse:
            self.feature_extraction = feature_extraction(False)
            self.attention_fuse = attention_fuse()
            self.ca_and_pred_g = cost_aggr_and_pred(self.num_groups, 32)

        # gwcnet-c
        if self.use_concat_volume and not self.use_attention_fuse:
            self.concat_channels = 24
            self.feature_extraction = feature_extraction(True, self.concat_channels)
            self.ca_and_pred_g = cost_aggr_and_pred(self.num_groups, 16)
            self.ca_and_pred_c = cost_aggr_and_pred(self.concat_channels * 2, 16)
            # self.cf_and_pred = cost_fuse_and_pred()
            self.dres = hourglass(32)
            self.classif = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                         Mish(),
                                         nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # gwcnet-ca
        if self.use_concat_volume and self.use_attention_fuse:
            self.concat_channels = 32
            self.feature_extraction = feature_extraction(True, self.concat_channels)
            self.attention_fuse = attention_fuse()
            self.ca_and_pred_g = cost_aggr_and_pred(self.num_groups, 32)
            self.ca_and_pred_c = cost_aggr_and_pred(self.concat_channels * 2, 16)
            self.dres = hourglass(48)
            self.classif = nn.Sequential(convbn_3d(48, 48, 3, 1, 1),
                                         Mish(),
                                         nn.Conv3d(48, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        _, _, h, w = left.size()
        # gwcnet-g
        if not self.use_concat_volume and not self.use_attention_fuse:
            features_left = self.feature_extraction(left)
            features_right = self.feature_extraction(right)
            gwc_volume = build_gwc_volume(torch.cat(features_left[0:3], dim=1), torch.cat(features_right[0:3], dim=1),
                                          self.maxdisp // 4, self.num_groups)
            outs, costs, preds = self.ca_and_pred_g(gwc_volume, self.maxdisp, h, w)
            return preds

        # gwcnet-a
        if not self.use_concat_volume and self.use_attention_fuse:
            features_left = self.feature_extraction(left)
            features_right = self.feature_extraction(right)
            features_left, features_right = self.attention_fuse(features_left, features_right)
            gwc_volume = build_gwc_volume(torch.cat(features_left[0:3], dim=1), torch.cat(features_right[0:3], dim=1),
                                          self.maxdisp // 4, self.num_groups)
            outs, costs, preds = self.ca_and_pred_g(gwc_volume, self.maxdisp, h, w)
            return preds

        # gwcnet-c
        if self.use_concat_volume and not self.use_attention_fuse:
            features_left = self.feature_extraction(left)
            features_right = self.feature_extraction(right)
            gwc_volume = build_gwc_volume(torch.cat(features_left[0:3], dim=1), torch.cat(features_right[0:3], dim=1),
                                          self.maxdisp // 4, self.num_groups)
            concat_volume = build_concat_volume(features_left[3], features_right[3], self.maxdisp // 4)

            outs_g, costs_g, preds_g = self.ca_and_pred_g(gwc_volume, self.maxdisp, h, w)
            outs_c, costs_c, preds_c = self.ca_and_pred_c(concat_volume, self.maxdisp, h, w)

            # preds = self.cf_and_pred(costs_g, costs_c, self.maxdisp, h, w)
            out = self.dres(torch.cat([outs_g[-1], outs_c[-1]], dim=1))
            cost = self.classif(out)
            cost = F.interpolate(cost, [self.maxdisp, h, w], mode='trilinear')
            cost = torch.squeeze(cost, 1)
            pred = F.softmax(cost, dim=1)
            pred = disparity_regression(pred, self.maxdisp)

            preds = []
            preds += preds_g + preds_c
            preds.append(pred)

            return preds

        # gwcnet-ca
        if self.use_concat_volume and self.use_attention_fuse:
            features_left = self.feature_extraction(left)
            features_right = self.feature_extraction(right)

            features_left, features_right = self.attention_fuse(features_left, features_right)

            gwc_volume = build_gwc_volume(torch.cat(features_left[0:3], dim=1), torch.cat(features_right[0:3], dim=1),
                                          self.maxdisp // 4, self.num_groups)
            concat_volume = build_concat_volume(features_left[3], features_right[3], self.maxdisp // 4)

            outs_g, costs_g, preds_g = self.ca_and_pred_g(gwc_volume, self.maxdisp, h, w)
            outs_c, costs_c, preds_c = self.ca_and_pred_c(concat_volume, self.maxdisp, h, w)

            # preds = self.cf_and_pred(costs_g, costs_c, self.maxdisp, h, w)
            out = self.dres(torch.cat([outs_g[-1], outs_c[-1]], dim=1))
            cost = self.classif(out)
            cost = F.interpolate(cost, [self.maxdisp, h, w], mode='trilinear')
            cost = torch.squeeze(cost, 1)
            cost = F.softmax(cost, dim=1)
            pred = disparity_regression(cost, self.maxdisp)

            with torch.no_grad():
                info_enty = torch.log(cost)
                info_enty[info_enty == float('-inf')] = 0.0
                info_enty = -cost * info_enty
                info_enty = torch.sum(info_enty, 1)

            pred_refine = self.refine(info_enty[:, None], pred[:, None], left).squeeze(dim=1)
            # print(info_enty.max(), info_enty.min())

            preds = []
            preds += preds_g + preds_c

            if self.training:
                preds.append(pred)
                preds.append(pred_refine)
            else:
                preds.append(pred_refine)

            return preds, [pred]


def GwcNet_G(d):
    return GwcNet(d, use_concat_volume=False, use_attention_fuse=False)

def GwcNet_A(d):
    return GwcNet(d, use_concat_volume=False, use_attention_fuse=True)

def GwcNet_C(d):
    return GwcNet(d, use_concat_volume=True, use_attention_fuse=False)

def GwcNet_CA(d):
    return GwcNet(d, use_concat_volume=True, use_attention_fuse=True)
