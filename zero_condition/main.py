#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 23:43:43 2023

@author: yanglei
"""
import cv2
import boxx
from boxx.ylth import *
import math
import torch
import random
import torch.nn.functional as F
import threading

with boxx.inpkg(), boxx.impt(".."):
    from ddn_utils import *
    import torch_utils, dnnlib


def topk_sample(probs, topk):
    if isinstance(topk, float):
        topk = max(1, int(round(len(probs) * topk)))
    module = torch if isinstance(probs, torch.Tensor) else np
    args = module.argsort(
        probs,
    )
    idx_k = int(random.choice(args[-topk:]))
    return idx_k


class L2Sampler:
    """only for 1 data, not whole batch, need to BatchedGuidedSampler(sampler)"""

    def __init__(self, target):
        self.raw = target  # guided raw format, here is (3, h, w) of torch.cuda.FloatTensor  [-1, 1]
        self.target = target[None]  # process target in __init__

    def __call__(self, dic):
        rgbs = dic["rgbs"]  # K RGB outputs for 1 data, not whole batch
        k, c, h, w = rgbs.shape
        resized = nn.functional.interpolate(self.target, (h, w), mode="area")
        probs = nn.functional.softmax(-((rgbs - resized) ** 2).mean([-1, -2, -3]), 0)
        return dict(
            probs=probs,  # K probabilities for sampling
            idx_k=int(probs.argmax()),  # sampled index
            condition0=self.target,
            condition_source0=self.raw,
        )


class L1Sampler(L2Sampler):
    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        resized = nn.functional.interpolate(self.target, (h, w), mode="area")
        probs = nn.functional.softmax(
            -(torch.abs(rgbs - resized)).mean([-1, -2, -3]), 0
        )
        return dict(
            probs=probs,
            idx_k=int(probs.argmax()),
            condition0=self.target,
            condition_source0=self.raw,
        )


def get_mask_s(shape=None):
    canvas = np.zeros((512, 512), dtype=np.uint8)

    # 定义字体类型和大小
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 15

    # 使用cv2.getTextSize获取文字宽度和高度
    thickness = 50
    text_size = cv2.getTextSize("S", font, font_scale, thickness=thickness)[0]

    # 计算文本的基线开始位置，以便让"S"出现在中心
    text_x = (canvas.shape[1] - text_size[0]) // 2 + thickness // 2
    text_y = (canvas.shape[0] + text_size[1]) // 2 - thickness // 5

    # 在画布上绘制文本
    canvas = cv2.putText(
        canvas, "S", (text_x, text_y), font, font_scale, (255), thickness, cv2.LINE_AA
    )
    if shape:
        canvas = cv2.resize(canvas, shape[::-1])
    mask_s = torch.from_numpy(canvas > 128)
    return mask_s


class L2MaskedSampler:
    def __init__(self, target, mask=0):
        self.raw = target
        h, w = target.shape[-2:]
        if isinstance(mask, int):
            mask_ = torch.zeros((h, w)) > 0
            if mask == 0:  # r
                mask_[:, : w // 2] = 1
            if mask == 1:  # d
                mask_[: h // 2] = 1
            if mask == 2:  # u
                mask_[h // 2 :] = 1
            if mask == 3:  # checkboard
                mask_[: h // 2, : w // 2] = 1
                mask_[h // 2 :, w // 2 :] = 1
            if mask == 4:  # ~eyes
                mask_[h * 2 // 5 : h * 5 // 9] = 1
            if mask == 5:  # ~cross
                mask_[h * 2 // 5 : h * 5 // 9] = 1
                mask_[:, w * 2 // 5 : w * 3 // 5] = 1
            if mask == 6:  # ~center crop
                mask_[
                    h // 4 : h * 3 // 4,
                    w // 4 : w * 3 // 4,
                ] = 1
            if mask == 7:  # ~ mask s
                mask_ = get_mask_s((h, w))
            if mask == 8:  # mask s
                mask_ = ~get_mask_s((h, w))
            if mask == 9:  # eyes
                mask_[h * 2 // 5 : h * 5 // 9] = 1
                mask_ = ~mask_
            mask = mask_
        self.mask = mask.cuda()
        self.target = target[None] * self.mask

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        resized = nn.functional.interpolate(self.target, (h, w), mode="area")
        mask_resized = nn.functional.interpolate(
            self.mask[None, None].float(), (h, w), mode="area"
        )  # [0,0]
        loss = torch.abs(rgbs - resized)  # l1
        loss = (rgbs - resized) ** 2  # l2
        probs = nn.functional.softmax(
            -((loss) * mask_resized).sum([-1, -2]).mean(-1)
            / (mask_resized.sum() + eps),
            0,
        )
        return dict(
            probs=probs,
            idx_k=topk_sample(probs, 2),
            condition0=self.target,
            condition_source0=self.raw,
        )

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        hh, ww = self.target.shape[-2:]
        rgbs = nn.functional.interpolate(rgbs, (hh, ww), mode="area")

        # loss = torch.abs(rgbs-self.target) # l1
        loss = (rgbs - self.target) ** 2  # l2
        mask_resized = self.mask[None, None].float()
        probs = nn.functional.softmax(
            -((loss) * mask_resized).sum([-1, -2]).mean(-1)
            / (mask_resized.sum() + eps),
            0,
        )
        return dict(
            probs=probs,
            idx_k=topk_sample(probs, 2),
            condition0=self.target,
            condition_source0=self.raw,
        )


class NoiseSampler(L2Sampler):
    def __init__(self, target, noise_rate=0.53):
        self.raw = target
        self.target = target[None] + torch.randn_like(target)[None] * noise_rate

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        resized = nn.functional.interpolate(self.target, (h, w), mode="area")
        probs = nn.functional.softmax(-((rgbs - resized) ** 2).mean([-1, -2, -3]), 0)
        return dict(
            probs=probs,
            idx_k=topk_sample(probs, 2),
            condition0=self.target,
            condition_source0=self.raw,
        )


class LowBitSampler(L2Sampler):
    def __init__(self, target, brightnessn=4):
        self.raw = target
        self.f = lambda x: ((x) * brightnessn / 2).round() * 2 / brightnessn
        self.target = self.f(target[None])

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        resized = nn.functional.interpolate(self.target, (h, w), mode="area")
        probs = nn.functional.softmax(-((rgbs - resized) ** 2).mean([-1, -2, -3]), 0)
        return dict(
            probs=probs,
            idx_k=topk_sample(probs, 2),
            condition0=self.target,
            condition_source0=self.raw,
        )

    def __call__2(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        # resized = nn.functional.interpolate(self.target, (h, w), mode="area")
        hh, ww = self.target.shape[-2:]
        resized = self.target
        rgbs = nn.functional.interpolate(rgbs, (hh, ww), mode="nearest")
        probs = nn.functional.softmax(
            -((self.f(rgbs) - resized) ** 2).mean([-1, -2, -3]), 0
        )
        return dict(
            probs=probs,
            idx_k=topk_sample(probs, 2),
            condition0=self.target,
            condition_source0=self.raw,
        )


class ColorfulSampler:
    def __init__(self, target):
        self.raw = target
        self.target = target[None].mean(-3, keepdims=True)

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        resized = nn.functional.interpolate(self.target, (h, w), mode="area")
        rgbs = rgbs.mean(-3, keepdims=True)
        probs = nn.functional.softmax(-((rgbs - resized) ** 2).mean([-1, -2, -3]), 0)

        return dict(
            probs=probs,
            idx_k=topk_sample(probs, 2),
            condition0=self.target,
            condition_source0=self.raw,
        )


class SuperResSampler:
    def __init__(self, target, shape=0.5):
        self.raw = target
        self.target = boxx.resize(target[None], shape, "area")

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        hh, ww = self.target.shape[-2:]
        resized = nn.functional.interpolate(rgbs, (hh, ww), mode="area")
        probs = nn.functional.softmax(
            -((self.target - resized) ** 2).mean([-1, -2, -3]), 0
        )
        return dict(
            probs=probs,
            idx_k=topk_sample(probs, 2),
            condition0=self.target,
            condition_source0=self.raw,
        )


class CannySampler:
    def __init__(self, target):
        self.raw = target
        self.f = lambda timg: cv2.Canny(t2rgb(timg), 100, 200)
        self.target = self.f(target)
        self.cache = {self.target.shape: self.target}

    def __call__(self, dic):
        rgbs = dic["rgbs"]

        k, h, w, c = rgbs.shape
        hh, ww = self.target.shape[-2:]
        rgbs = nn.functional.interpolate(rgbs, (hh, ww), mode="nearest")
        rgbs = (
            rgbs.permute(0, 2, 3, 1).cpu().numpy()
        )  # Change tensor shape to (k, h, w, c)

        # if (h, w) not in self.cache:
        #     raw_resized = nn.functional.interpolate(
        #                self.raw[None], (h, w), mode="area"
        #             )[0]
        #     self.cache[(h, w)] = self.f(raw_resized)
        # target_resized=self.cache[(h, w)]
        target_resized = self.target

        resized_imgs = []
        for img in rgbs:
            resized_img = self.f(img)
            resized_imgs.append(resized_img)

        resized_imgs = np.stack(
            resized_imgs, axis=0
        )  # Stacking the images into one numpy array
        # if w == 64:
        #     g()/0
        if not increase("edge show") % 8:
            show(resized_imgs[:1], rgbs[:1], t2rgb)
        # probs = -(np.abs(resized_imgs - target_resized).clip(0.)).mean(-1).mean(-1)
        # probs = -(np.abs(resized_imgs[target_resized>0]).clip(0.)).mean(-1)#.mean(-1)
        probs = resized_imgs[:, target_resized > 0].mean(-1) + (
            1 - resized_imgs[:, target_resized < 0.5].mean(-1)
        )
        # probs = - probs
        if h <= 16:
            probs += resized_imgs.mean(-1).mean(-1)
        g()
        # probs = F.softmax(torch.tensor((target_resized[None] == resized_imgs).sum(-1).sum(-1).astype(np.float32)), 0)
        return dict(
            probs=probs,
            idx_k=topk_sample(probs, 2),
            condition0=self.target,
            condition_source0=self.raw,
        )


class EdgeSampler:
    def __init__(self, target):
        self.raw = target
        self.sobel_kernel_horizontal = (
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])[
                None, None
            ]
            .requires_grad_(False)
            .cuda()
        )
        self.sobel_kernel_vertical = (
            torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])[
                None, None
            ]
            .requires_grad_(False)
            .cuda()
        )

        self.target = self.sobel_operator(target[None])
        self.cache = {self.target.shape: self.target}

    def sobel_operator(self, img, h=None):
        grey = img.mean(-3, keepdim=True)
        if h is None:
            h, w = grey.shape[-2:]

        output_horizontal = F.conv2d(grey, self.sobel_kernel_horizontal)
        output_vertical = F.conv2d(grey, self.sobel_kernel_vertical)

        magnitude = torch.sqrt(output_horizontal**2 + output_vertical**2)
        thre = 1.5
        thre = 1.0
        edge = magnitude
        if h < 50:
            thre = 2
        if h < 20:
            thre = 2.1
        # edge = (magnitude > thre).float() * 2 - 1
        return edge

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        # if h < 5:
        #     return dict(probs=torch.rand(k))
        hh, ww = self.target.shape[-2:]
        target_resized = self.target
        if (h, w) not in self.cache and h > 600:
            # raw_resized = nn.functional.interpolate(
            #     self.raw[None], (h, w), mode="area"
            # )
            # target_resized = self.sobel_operator(raw_resized)

            # self.cache[(h, w)] = target_resized
            show(target_resized, rgbs[0], t2rgb, self.sobel_operator(rgbs[:1]))

        # target_resized = self.cache[(h, w)]
        rgbs = nn.functional.interpolate(rgbs, (hh + 2, ww + 2), mode="bilinear")
        resized_imgs = self.sobel_operator(rgbs, h=h)

        if 0:
            # 定义一个3x3的全一矩阵作为卷积核，大小为9
            kernel = torch.ones((1, 1, 3, 3), device="cuda:0")
            # kernel = torch.ones((1, 1, 3, 3), device='cuda:0')

            # 在 target_resized 上执行卷积操作，计算出每个位置的周围有多少个值等于其自身的值
            convolved_imgs = F.conv2d(resized_imgs, kernel, padding=1)

            # 将 target_resized 扩展为与 convolved_imgs 相同的维度以便比较
            target_resized_expanded = target_resized.expand_as(convolved_imgs)

            # 检查卷积后的图像中哪些位置的值等于 target_resized 中的对应值
            masks = convolved_imgs == target_resized_expanded

        # masks = target_resized == resized_imgs
        # probs = (masks).float().mean(-1).mean(-1).mean(-1)
        # probs=resized_imgs[:,:,target_resized[0,0]>0].mean(-1) + (1 - resized_imgs[:,:,target_resized[0,0]<0].mean(-1))
        # probs=  probs[:,0]
        probs = (
            -(torch.abs(resized_imgs - target_resized).clip(0.0))
            .float()
            .mean(-1)
            .mean(-1)
            .mean(-1)
        )
        # probs = - probs
        idx_k = topk_sample(probs, 2)
        g()
        if not increase("edge show") % 8:
            show(resized_imgs[:1], rgbs[:1], t2rgb)
        return dict(
            probs=probs,
            idx_k=idx_k,
            condition0=self.target,
            condition_source0=resized_imgs[0],
        )


def tensor_to_imagenet_format(rgbs):
    img_tensor = (rgbs + 1) / 2
    # img_tensor = torch.nn.functional.interpolate(img_tensor, size=(224, 224), mode="nearest")
    img_tensor = torch.nn.functional.interpolate(
        img_tensor, size=(64, 64), mode="nearest"
    )

    def normalize(tensor, mean, std):
        return (tensor - mean) / std

    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).cuda()
    img_tensor_norm = normalize(img_tensor, mean, std)
    return img_tensor_norm


class StyleResNet(torchvision.models.ResNet):
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x.view(b, -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    @staticmethod
    def build_with_pretrain():
        pretrain = torchvision.models.resnet18(
            weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT
        ).state_dict()
        model = StyleResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
        model.load_state_dict(pretrain)
        return model


class StyleTransferResnet:
    def __init__(self, target):
        self.model = StyleResNet.build_with_pretrain().eval().cuda()
        self.raw = target
        self.target = self.f(target[None])

    def f(self, rgbs):
        with torch.no_grad():
            return self.model(tensor_to_imagenet_format(rgbs))

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        feats = self.f(rgbs)
        probs = nn.functional.softmax(-((self.target - feats) ** 2).mean([-1]), 0)
        return dict(
            probs=probs,
            idx_k=topk_sample(probs, 2),
            condition0=self.target,
            condition_source0=self.raw,
        )


class StyleVGG(torchvision.models.VGG):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.features[:1](x)
        # sampler.model.features
        # g()/0
        return x.view(b, -1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @staticmethod
    def build_with_pretrain():
        pretrain = torchvision.models.vgg11_bn(
            weights=torchvision.models.vgg.VGG11_BN_Weights.DEFAULT
        ).state_dict()
        model = StyleVGG(
            torchvision.models.vgg.make_layers(
                [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
                batch_norm=True,
            ),
        )
        model.load_state_dict(pretrain)
        return model


class StyleTransferVGG(StyleTransferResnet):
    def __init__(self, target):
        self.model = StyleVGG.build_with_pretrain().eval().cuda()
        self.raw = target
        self.target = self.f(target[None])


StyleTransfer = StyleTransferResnet
# StyleTransfer = StyleTransferVGG


class CifarSampler:
    def __init__(self, target=None, entropy=False):
        with impt("../../asset/resnet_cifar_zhk"):
            from cifar_pretrain import CifarPretrain
        self.model = CifarPretrain()  # rgbs => (k, 10) feat
        self.target = target
        self.entropy = entropy

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        # rgbs = torch.nn.functional.interpolate(rgbs, size=(32, 32), mode="bilinear")
        rgbs = torch.nn.functional.interpolate(rgbs, size=(32, 32), mode="nearest")
        class_probs = nn.functional.softmax(self.model(rgbs), -1)
        target = self.target
        if target is None:
            if "idx_gen" in dic:
                target = int(dic["idx_gen"] % 10)
                # print(target, dic["idx_gen"])
        topk = 0.1
        if self.entropy:
            target = None
            topk = 0.1
        if target is None:
            probs = class_probs.max(1)[0]
        else:
            probs = class_probs[:, target]

        idx_k = topk_sample(probs, topk)
        mg()
        # if random.random() < 1.5 and h<=9:
        # if random.random() < 1.5 and h<=7:
        #     idx_k = random.randint(0, len(rgbs)-1)
        # if not increase("edge show") % 28:
        #     show(rgbs[:1], t2rgb)
        # print(target, dic["idx_gen"])
        # print(classi2name[target],target, class_probs.max(1)[1][:5])
        return dict(
            probs=probs,
            idx_k=idx_k,
            condition0=target,
        )


def FaceRecognizeSampler(target):
    with impt("../../asset/face_recon_sampler"):
        from face_recon_sampler import FaceRecognizeSampler
    return FaceRecognizeSampler(target)


class CLIPSampler:
    model_in_memory = None
    lock = threading.Lock()

    def __init__(self, target, keep_model_in_memory=False):
        import clip

        self.raw = target
        model_name = "ViT-B/32"  # 338M  # 0.07384706
        # model_name = 'RN50' # 244M  # 0.09297562
        # model_name = 'RN50x64' # 1.26G
        device = "cuda"
        # self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # RuntimeError: expected scalar type Float but found Half
        self.dtype = torch.float32
        with CLIPSampler.lock:
            if keep_model_in_memory and CLIPSampler.model_in_memory is not None:
                self.model = CLIPSampler.model_in_memory
            else:
                self.model, transform = clip.load(model_name, device=device)
                self.model = self.model.to(device)  #
                # looks like clip.load already set float16
                # if do .to(float32)  will slow 2x times.
                if keep_model_in_memory:
                    CLIPSampler.model_in_memory = self.model
        with torch.no_grad():
            tokens = clip.tokenize([target]).to(device)
            self.target = self.model.encode_text(tokens)

    def __call__(self, dic):
        rgbs = dic["rgbs"]  # .to(self.dtype)
        dtype = rgbs.dtype
        device = rgbs.device
        mean = (
            torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            .reshape(
                3,
                1,
                1,
            )
            .to(device)
        )
        std = (
            torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            .reshape(
                3,
                1,
                1,
            )
            .to(device)
        )
        rgbs = ((rgbs + 1) / 2 - mean) / std

        # rgbs = nn.functional.interpolate(rgbs, (224, 224), mode="area")
        # rgbs = nn.functional.interpolate(rgbs, (224, 224), mode='bilinear', align_corners=False)
        rgbs = nn.functional.interpolate(rgbs, (224, 224), mode="bicubic")

        with torch.no_grad():
            num_minibatches = math.ceil(rgbs.size(0) / 8)
            minibatches = torch.chunk(rgbs, num_minibatches)
            image_features = []
            for batch in minibatches:
                image_features.append(self.model.encode_image(batch))
            image_features = torch.cat(image_features, dim=0)

            text_features = self.target

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # similarity = (100 * image_features @ text_features.T).softmax(dim=0)

        # probs = ((similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-6)).squeeze()
        probs = (100 * image_features @ text_features.T)[:, 0]

        # 后面的 idx_k condition_source0 condition0 可以不返回, 但推荐返回, 方便看效果
        mg()
        return dict(
            probs=probs,
            idx_k=topk_sample(probs, 1 if len(probs) == 2 else 2),
            condition_source0=self.raw,
            condition0=self.target,
        )


class ReconstructionDatasetSampler:
    def __init__(
        self,
        dataset=None,
    ):
        """
        dataset[k] will return one h,w,3 RGB numpy image
        """
        if dataset is None:

            class CIFAR10Uint8(torchvision.datasets.cifar.CIFAR10):
                def __getitem__(self, idx):
                    pil, label = super().__getitem__(idx % 10000)
                    return np.array(pil)

            dataset = CIFAR10Uint8(
                os.path.expanduser("~/dataset"),
                train=False,
                # download=True,
            )
        self.dataset = dataset

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        if "idx_gen" in dic:
            data_idx = int(dic["idx_gen"])
        else:
            data_idx = random.randint(0, len(self.dataset) - 1)
        if "sampler_context" not in dic:
            img = self.dataset[data_idx % len(self.dataset)]
            if isinstance(img, tuple):
                img = img[0]
            if img.shape[-1] > 3 and img.shape[-3] <= 3:
                img = img.transpose(1, 2, 0)
            # show-img
            context = dict(
                target=uint8_to_tensor(img)[None],
                data_idx=data_idx,
            )
        else:
            context = dic["sampler_context"]

        resized = nn.functional.interpolate(context["target"], (h, w), mode="area")
        probs = nn.functional.softmax(-((rgbs - resized) ** 2).mean([-1, -2, -3]), 0)
        return dict(
            probs=probs,
            idx_k=int(probs.argmax()),
            condition0=context["target"],
            # condition_source0=target,
            sampler_context=context,
        )


class BatchedGuidedSampler:
    def __init__(self, sampler=None):
        sampler = sampler or (lambda d: dict(probs=[1.0] + [0] * (len(d["rgbs"]) - 1)))
        self.sampler = sampler

    def __call__(self, d):
        outputs = d["output"]
        b, k, c, h, w = outputs.shape
        dics = []
        for batchi in range(b):
            input_dic = dict(rgbs=outputs[batchi])
            if "idx_gens" in d:
                input_dic["idx_gen"] = d["idx_gens"][batchi]
            if "sampler_contexts" in d:
                input_dic["sampler_context"] = d["sampler_contexts"][batchi]
            dic = self.sampler(input_dic)
            dics.append(dic)
        if "idx_k" in dic:
            idx_ks = [dic["idx_k"] for dic in dics]
        elif "probs" in dic:
            idx_ks = npa([np.argmax(dic["probs"]) for dic in dics])
        if "condition_source0" in dic:
            d["condition_source0"] = [dic["condition_source0"] for dic in dics]
        if "condition0" in dic:
            d["condition0"] = [dic["condition0"] for dic in dics]
        if "sampler_context" in dic:
            d["sampler_contexts"] = [dic["sampler_context"] for dic in dics]
        return npa(idx_ks)


class MultiGuidedSampler:
    def __init__(self, sampler_to_weight):
        self.sampler_to_weight = sampler_to_weight

    def __call__(self, d):
        s2w = self.sampler_to_weight
        sn = len(s2w)
        outputs = d["output"]
        b, k, c, h, w = outputs.shape
        probsd = {}
        weighted_argsortd = {}
        for si, sampler in enumerate(s2w):
            dics = []
            for batchi in range(b):
                input_dic = dict(rgbs=outputs[batchi])
                if "idx_gens" in d:
                    input_dic["idx_gen"] = d["idx_gens"][batchi]
                if "sampler_contexts" in d:
                    input_dic["sampler_context"] = d["sampler_contexts"][si][batchi]
                dic = sampler(input_dic)
                dics.append(dic)
            probsd[sampler] = [npa(dic["probs"]) for dic in dics]
            weighted_argsortd[sampler] = [
                np.argsort(npa(dic["probs"])).argsort() * s2w[sampler] for dic in dics
            ]  # si: b, k
            if "condition_source0" in dic:
                d["condition_source0"] = d.get("condition_source0", {})
                d["condition_source0"][si] = [dic["condition_source0"] for dic in dics]
            if "condition0" in dic:
                d["condition0"] = d.get("condition0", {})
                d["condition0"][si] = [dic["condition0"] for dic in dics]
            if "sampler_context" in dic:
                d["sampler_contexts"] = d.get("sampler_contexts", {})
                d["sampler_contexts"][si] = [dic["sampler_context"] for dic in dics]
        weighted_argsort = npa(list(weighted_argsortd.values()))  # sn, b, k
        bk = weighted_argsort.sum(0)  # b, k
        # idx_ks = bk.argmax(1)  # b
        idx_ks = [topk_sample(prob, 2) for prob in bk]
        # print(dic, idx_ks)
        # g()/0
        return npa(idx_ks)


classi2name = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}
if __name__ == "__main__":
    datapath = "../../asset/outputs-cifar.pt"
    datapath = "../../asset/outputs-ffhq64.pt"
    train_imgs = torch.load(datapath).cuda()

    pklp = "../../asset/v15_00022-cifar10-blockn32_outputk64_chain.dropout0.05_fp32-shot-200000.pkl"
    pklp = "../../asset/v15-00035-cifar10-32x32-cifar_blockn32_outputk64_chain.dropout0.05_fp32_goon.v15.22-shot-087808.pkl"

    # pklp = "../../asset/v15-00040-cifar10-32x32-cifar_fp32_blockn32_outputk512_chain.dropout0.05_transfer.00035.kimg100352-shot-007526.pkl"

    if "ffhq" in datapath:
        pklp = "../../asset/v13_new.setting-00000-ffhq64-fp16-dropout0-200000.pkl"
        pklp = "../../asset/v15-00023-ffhq-64x64-blockn64_outputk64_chain.dropout0.05-shot-117913.pkl"
        # pklp = "../../asset/v17-00016-ffhq-64x64-outputk2_blockn64_chain.dropout0-shot-082790.pkl"
        # pklp = "../../asset/v15-00018-ffhq-64x64-blockn64_outputk512_chain.dropout0.05-shot-117913.pkl"

    net = sys._getframe(3 if boxx.sysi.gui else 0).f_globals.get("net")
    if net is None:
        print("load net....")
        net = load_net(pklp)

    train_imgs_xflip = tht(npa(train_imgs)[..., ::-1].copy()).cuda()
    train_idx = 1
    if 1:
        pass
    for train_idx in range(3):
        target = train_imgs[train_idx]
        target = train_imgs_xflip[train_idx]
        # save_data(d["predict"], "/tmp/predict-as-target")
        # target = load_data("/tmp/predict-as-target")[0]
        dt = dict(target=target[None])

        # for i in range(3):
        pass
        sampler = L2Sampler(target)
        # sampler = L1Sampler(target)
        # sampler = ColorfulSampler(target)
        # sampler = SuperResSampler(target, 1/8)
        # sampler = NoiseSampler(target)
        # sampler = LowBitSampler(target)
        maski = 9
        # for maski in range(10):
        sampler = L2MaskedSampler(target, maski)
        # Canny 和 Edge 效果不好
        # sampler = CannySampler(target, )
        # sampler = EdgeSampler(target, )
        # sampler = StyleTransfer(target, )
        classi = None
        classi = 8

        # for classi in range(10):
        # sampler = CifarSampler(classi)
        # sampler = CifarSampler(entropy=True)
        # sampler = FaceRecognizeSampler(target)
        # print(sampler(dict(rgbs=torch.cat([train_imgs, train_imgs_xflip])))["probs"].tolist())
        """
        # 重建的人脸做 target 识别没问题
[0.0014740412589162588, 0.00017411627050023526, 0.6346914768218994, 0.007211570627987385, 0.00024585213395766914, 0.3562029302120209]
        """

        # target_ = "black man"
        target_ = "man wearing sunglasses"
        # target_ = "woman wearing sunglasses"
        # target_ = "a young blonde woman laughing heartily"
        # target_ = "a woman with a somewhat unhappy expression and a pout"
        # target_ = "head portrait, a young blonde woman laughing under blue sky and green grass"
        # target_ = "Donald Trump"

        # sampler = CLIPSampler(target_)

        # sampler = ReconstructionDatasetSampler()

        # print(sampler(dict(rgbs=train_imgs_xflip))["probs"])

        batch_sampler = BatchedGuidedSampler(sampler)
        # samplerd = {
        #     L2Sampler(train_imgs[train_idx]): 1,
        #     L2Sampler(train_imgs_xflip[(train_idx + 1) % 3]): 1,
        # }
        # samplerd = {
        #     ColorfulSampler(train_imgs[train_idx]): .25,
        #     SuperResSampler(train_imgs_xflip[(train_idx + 1) % 3], 1 / 8): 0.25,
        #     CLIPSampler("wearing sunglasses"):.5,
        # }
        # samplerd = {
        #     L2MaskedSampler(target, 9):.25,
        #     CLIPSampler("wearing sunglasses"):.75,
        # }
        # batch_sampler = MultiGuidedSampler(samplerd)
        d = dict(sampler=batch_sampler)
        # d["idx_gens"] = [0]
        d = net(d)
        # tree-d
        if isinstance(sampler, CifarSampler):
            print(
                classi,
                classi2name.get(classi, "None"),
                "class:",
                sampler.model(d["predict"]).max(1)[1].item(),
            )
        if boxx.sysi.gui:
            showd(d, 1, figsize=(8, 5) if target.shape[-1] >= 40 else (4, 3))
        else:
            tree(d, deep=1)
            # shows(d["predict"], t2rgb, png=True)

        # d=net(dict(batch_size=6));showd(d,1);sampler.model(d["predict"]).max(1)
