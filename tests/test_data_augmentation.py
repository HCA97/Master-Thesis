import torchvision
from torchvision import transforms
import torch as th
# from torchvision import utils
from scripts.dataloader import *

brightness = [1, 1.1]
contrast = [1, 2]
saturation = [1, 1]
hue = [-0.2, 0.5]

if __name__ == "__main__":

    potsdam_dir = "../potsdam_data/potsdam_cars"

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
                                         brightness=brightness, contrast=contrast, hue=hue, saturation=saturation),
                                    transforms.Normalize([0.5], [0.5])])
    # transform6 = transforms.Compose([transforms.Resize((32+5, 64+5)),
    #                                 transforms.RandomCrop((32, 64)),
    #                                 transforms.ColorJitter(
    #                                     contrast=contrast, hue=hue),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize([0.5], [0.5])])
    transforms = [transform1, transform2, transform3,
                  transform4, transform5, transform6]
    names = ["brightness", "contrast", "saturation",
             "hue", "default", "brightness_contrast"]

    for name, transform in zip(names, transforms):
        postdamdataset = PostdamCarsDataModule(
            potsdam_dir, transform=transform)
        postdamdataset.setup()

        dataloader = postdamdataset.train_dataloader()
        imgs = next(iter(dataloader))[0]
        grid = torchvision.utils.make_grid(
            imgs, 8, normalize=True, range=(-1, 1))
        torchvision.utils.save_image(grid, f"figs/test_{name}.png")
