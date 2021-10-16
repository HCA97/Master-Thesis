# import torch
# import lpips
# import time
# from scripts.utility import fid_score
# # loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
# # closer to "traditional" perceptual loss, when used for optimization
# loss_fn_vgg = lpips.LPIPS(net='vgg')

# # image should be RGB, IMPORTANT: normalized to [-1,1]
# img0 = torch.zeros(1024, 3, 32, 64)
# img1 = torch.zeros(1024, 3, 32, 64)

# s = time.time()
# d = fid_score(img0, img1, device="cpu", layer_idx=30, use_bn=False)
# e = time.time()
# print(e-s)
# print(d)
# # print(img0.device, img1.device)

# # s = time.time()
# # d = loss_fn_vgg(img0, img1)
# # e = time.time()
# # print(e-s)
# # print(d)

# import torch as th
# from scripts import *


# # def slerp(a, b, t):
# #     a = a / a.norm(dim=-1, keepdim=True)
# #     b = b / b.norm(dim=-1, keepdim=True)
# #     d = (a * b).sum(dim=-1, keepdim=True)
# #     p = t * th.acos(d)
# #     c = b - d * a
# #     c = c / c.norm(dim=-1, keepdim=True)
# #     d = a * th.cos(p) + c * th.sin(p)
# #     d = d / d.norm(dim=-1, keepdim=True)
# #     return d


# # gan = GAN.load_from_checkpoint(
# #     "/home/hca/Documents/Master/Thesis/Master-Thesis/experiments/baseline/lightning_logs/version_0/checkpoints/epoch=449.ckpt")
# # gan.eval()
# # print(perceptual_path_length(gan.generator,
# #       n_samples=1024, epsilon=1e-4, use_slerp=True))

# z1_ = th.normal(0, 1, size=(100,),
#                 device="cpu")
# z2_ = th.normal(0, 1, size=(100,),
#                 device="cpu")

# ratio = np.random.uniform(0, 1)
# # z1 = slerp(z1_, z2_, t=ratio)
# # z2 = slerp(z1_, z2_, t=1 - ratio)
# z1 = interpolate(z1_, z2_, ratio, True)
# z2 = interpolate(z2_, z1_, 1 - ratio, True)
# print(th.allclose(z1, z2))
# # print(z1, z2)

# #

import torchvision

a = torchvision.models.vgg16_bn(pretrained=True).features
print(a)
