from torchsummary import summary

from scripts.models.networks import *

# img_size = (3, 64, 128)
img_size = (3, 32, 64)
generator_params = {"n_layers": 4, "init_channels": 512,
                    "bn_mode": "default", "use_spectral_norm": True}
discriminator_params = {"base_channels": 32, "n_layers": 4,
                        "bn_mode": "default", "use_spectral_norm": True}

# net = RefinerNet(img_size[0], **generator_params).cuda()
# print(net)
# print(summary(net, img_size))
# a = net.forward(th.ones((1, 3, 32, 64), device="cuda:0"))
# # print(a)
# s = net.__str__()
# s1 = """UnetGenerator(
#   (down): ModuleList(
#     (0): ConvBlock(
#       (conv_block): ModuleList(
#         (0): Conv2d(3, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (1): BatchNorm2d(256, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
#         (2): LeakyReLU(negative_slope=0.2, inplace=True)
#         (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (4): BatchNorm2d(256, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
#         (5): LeakyReLU(negative_slope=0.2, inplace=True)
#       )
#     )
#     (1): ConvBlock(
#       (conv_block): ModuleList(
#         (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (1): BatchNorm2d(512, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (4): BatchNorm2d(512, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
#         (5): ReLU(inplace=True)
#       )
#     )
#   )
#   (bottle_neck): ConvBlock(
#     (conv_block): ModuleList(
#       (0): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (1): BatchNorm2d(1024, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU(inplace=True)
#       (3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (4): BatchNorm2d(1024, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU(inplace=True)
#     )
#   )
#   (up): ModuleList(
#     (0): Sequential(
#       (0): ConvBlock(
#         (conv_block): ModuleList(
#           (0): Conv2d(1536, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (1): BatchNorm2d(512, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
#           (2): ReLU(inplace=True)
#         )
#       )
#       (1): ConvBlock(
#         (conv_block): ModuleList(
#           (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (1): BatchNorm2d(512, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
#           (2): ReLU(inplace=True)
#           (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (4): BatchNorm2d(512, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
#           (5): ReLU(inplace=True)
#         )
#       )
#     )
#     (1): Sequential(
#       (0): ConvBlock(
#         (conv_block): ModuleList(
#           (0): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (1): BatchNorm2d(256, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
#           (2): ReLU(inplace=True)
#         )
#       )
#       (1): ConvBlock(
#         (conv_block): ModuleList(
#           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (1): BatchNorm2d(256, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
#           (2): ReLU(inplace=True)
#           (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (4): BatchNorm2d(256, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
#           (5): ReLU(inplace=True)
#         )
#       )
#     )
#   )
#   (final): ConvBlock(
#     (conv_block): ModuleList(
#       (0): Conv2d(256, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (1): Tanh()
#     )
#   )
# )"""
# print(s == s1)

# net = ResNetGenerator(img_size, **generator_params).cuda()
# print(summary(net, (100, 1, 1)))

# img = th.ones([1, 3, 32, 64]).cuda()
net = BasicDiscriminator(img_size, **discriminator_params).cuda()
print(net)
# x, h1, h2 = net.forward(img)
# print(x.shape, h1.shape, h2.shape)
print(summary(net, img_size))

print(generator_params)
net = BasicGenerator(img_size, **generator_params).cuda()
for l in net.conv_blocks:
    try:
        print(l.weight_u.size())
    except:
        pass
print(net)
print(summary(net, (100, 1, 1)))
