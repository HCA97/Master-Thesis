import torchvision
from torchvision import transforms
import torch as th
import PIL
# from torchvision import utils
from scripts.dataloader import *

brightness = [1, 1]
contrast = [1, 1.25]
saturation = [1, 1]
hue = [-0.1, 0.1]


# GaussianBlur
# RandomRotation
# ColorJitter
# RandomPerspective

if __name__ == "__main__":

    # potsdam_dir = "../potsdam_data/artificial_cars"
    potsdam_dir = "../potsdam_data/potsdam_cars_corrected"

    # transform_original = transforms.Compose([transforms.Resize((32, 64)), transforms.ToTensor(),
    #                                          transforms.RandomHorizontalFlip(
    #                                              p=0.5),
    #                                          transforms.RandomVerticalFlip(
    #                                              p=0.5),
    #                                          transforms.Normalize([0.5], [0.5])])
    # transform_data_aug = transforms.Compose([transforms.Resize((35, 67)),
    #                                          transforms.RandomCrop((32, 64)),
    #                                          transforms.ColorJitter(
    #                                              contrast=[1, 1.2], hue=[-0.2, 0.2]),
    #                                          transforms.RandomRotation(
    #     degrees=10, resample=PIL.Image.BILINEAR),
    #     transforms.RandomPerspective(
    #     distortion_scale=0.2),
    #     transforms.GaussianBlur(3),
    #     transforms.RandomHorizontalFlip(
    #                                              p=0.5),
    #     transforms.RandomVerticalFlip(
    #                                              p=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5], [0.5])])

    # transforms = [transform_original, transform_data_aug]
    # names = ["original", "data_aug"]

    # for name, transform in zip(names, transforms):
    #     postdamdataset = PostdamCarsDataModule(
    #         potsdam_dir, transform=transform)
    #     postdamdataset.setup()

    #     dataloader = postdamdataset.train_dataloader()
    #     imgs = next(iter(dataloader))[0]
    #     grid = torchvision.utils.make_grid(
    #         imgs, 8, normalize=True, range=(-1, 1), pad_value=1)
    #     torchvision.utils.save_image(grid, f"figs/test_fake_{name}.png")

    transform1 = transforms.Compose([transforms.Resize((32, 64)), transforms.ToTensor(),
                                    transforms.ColorJitter(
                                        brightness=brightness),
                                     transforms.Normalize([0.5], [0.5])])
    transform2 = transforms.Compose([transforms.Resize((32, 64)), transforms.ToTensor(),
                                    transforms.ColorJitter(
                                        contrast=contrast),
                                    transforms.Normalize([0.5], [0.5])])
    transform3 = transforms.Compose([transforms.Resize((32, 64)), transforms.ToTensor(),
                                     transforms.ColorJitter(
                                         saturation=saturation),
                                    transforms.Normalize([0.5], [0.5])])
    transform4 = transforms.Compose([transforms.Resize((32, 64)), transforms.ToTensor(),
                                     transforms.ColorJitter(
                                         hue=hue),
                                     transforms.Normalize([0.5], [0.5])])
    transform5 = transforms.Compose([transforms.Resize((32, 64)), transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])])
    transform6 = transforms.Compose([transforms.Resize((32, 64)), transforms.ToTensor(),
                                     transforms.ColorJitter(
                                         contrast=contrast, hue=hue),
                                    transforms.Normalize([0.5], [0.5])])

    transforms = [transform1, transform2, transform3,
                  transform4, transform5, transform6]
    names = ["brightness", "contrast", "saturation",
             "hue", "default", "contrast_hue"]

    for name, transform in zip(names, transforms):
        postdamdataset = PostdamCarsDataModule(
            potsdam_dir, transform=transform, shuffle=False)
        postdamdataset.setup()

        dataloader = postdamdataset.train_dataloader()
        imgs = next(iter(dataloader))[0]
        grid = torchvision.utils.make_grid(
            imgs, 8, normalize=True, range=(-1, 1), pad_value=1)
        torchvision.utils.save_image(grid, f"figs/test_{name}.png")
