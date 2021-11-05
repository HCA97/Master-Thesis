from torchsummary import summary

from scripts.models.networks import *

# img_size = (3, 64, 128)

# generator_params = {"n_layers": 4, "init_channels": 512,
#                     "bn_mode": "default", "act": "leakyrelu", "last_layer_kernel_size": 3}
# discriminator_params = {"base_channels": 64, "n_layers": 5, "heat_map": False,
#                         "bn_mode": "default", "use_spectral_norm": False}

# img_size = (3, 48, 96)
# discriminator_params = {"base_channels": 64, "n_res": 1, "kernel_size": 3}
# net = MultiDiscriminator(img_size, **discriminator_params).cuda()
# img1 = th.ones((10,) + img_size).cuda()
# x = net(img1, 1)
# for xx in x:
#     print(xx.shape)
# # print(summary(net, img_size))
# print(net)

# generator_params = {"n_layers": 4,
#                     "base_channels": 32,
#                     "padding_mode": "reflect"}
# img_size = (3, 32, 64)
# net = MUNITGEN(img_size, **generator_params).cuda()
# print(net)
# # img1 = th.ones((10,) + img_size).cuda()
# # img2 = th.ones((10,) + img_size).cuda()

# # [x11, x22, x12, x21, x121, x212], [c1, s1, c2, s2, c12, s12] = net(img1, img2)
# # # a, b = net(img1, img2)
# # for i in a:
# #     print(i.shape)
# # for i in b:
# #     print(i.shape)
# net2 = MUNITDecoder(img_size, **generator_params).cuda()
# net1 = MUNITEncoder(img_size, **generator_params).cuda()
# img = th.ones((10,) + img_size).cuda()
# c, s = net1(img)
# print(summary(net1, img_size))
# # print(c.shape, s.shape)
# # x = net2(c, s)
# # print(x.shape)
# print(summary(net2, [(128, 8, 16), (8,)]))

# print(net1)

# print(net2)
# discriminator_params = {"base_channels": 64, "n_layers": 4, "n_blocks": 1, "use_dropout": True,
#                         "bn_mode": "default", "use_spectral_norm": False}
# img_size = (3, 32, 64)
# net = ResNetDiscriminator(img_size, **discriminator_params).cuda()
# print(net)
# # # # # x, h1, h2 = net.forward(img)
# # # # # print(x.shape, h1.shape, h2.shape)
# print(summary(net, img_size))

# # # print(generator_params)
# generator_params = {"n_layers": 4, "init_channels": 512, "padding_mode": "reflect",
#                     "bn_mode": "default", "act": "leakyrelu"}
# net = ResNetGenerator(img_size, **generator_params).cuda()
# print(net)
# print(summary(net, (100, 1, 1)))
# # #

img_size = (3, 64, 128)
discriminator_params = {"base_channels": 64,
                        "padding_mode": "reflect",
                        "n_layers": 4,
                        "use_dropout": 0.5,
                        "bn_mode": "default"}
net = BasicDiscriminator(img_size, **discriminator_params).cuda()
print(net)
print(summary(net, img_size))
