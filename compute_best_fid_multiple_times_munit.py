import torchvision
from torchvision import transforms
from scripts import *

transform = transforms.Compose([transforms.Resize((32, 64)),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ColorJitter(
                                    hue=[-0.1, 0.1], contrast=[1, 1.25]),
                                transforms.Normalize([0.5], [0.5])])


potsdam_dir = "../potsdam_data/potsdam_cars_corrected"
munit_dir1 = "experiments/munit_version_2_larger_image_corrected/version_1/10000_cars_cropped"
munit_dir2 = "experiments/munit_version_2_larger_image_corrected_500/version_0/10000_cars_cropped"

#
# MULTIPLE FID SCORE
#

# ALL DATA
datasetmodule = PostdamCarsDataModule(potsdam_dir,
                                      data_dir2=munit_dir1,
                                      transform=transform,
                                      batch_size=1024)
datasetmodule.setup()
dataloader = datasetmodule.train_dataloader()

fid_multiple = []
for _ in range(5):
    real, refined = next(iter(dataloader))
    fid_multiple.append(fid_score(real, refined, device="cuda:0"))
print(f"FID MIN - {np.min(fid_multiple)} | FID MAX - {np.max(fid_multiple)} | FID Mean - {np.mean(fid_multiple)}")

# LIMITED DATA
datasetmodule = PostdamCarsDataModule(potsdam_dir,
                                      data_dir2=munit_dir2,
                                      transform=transform,
                                      batch_size=1024)
datasetmodule.setup()
dataloader = datasetmodule.train_dataloader()

fid_multiple = []
for _ in range(5):
    real, refined = next(iter(dataloader))
    fid_multiple.append(fid_score(real, refined, device="cuda:0"))
print(f"FID MIN - {np.min(fid_multiple)} | FID MAX - {np.max(fid_multiple)} | FID Mean - {np.mean(fid_multiple)}")


#
# 30 RANDOMLY GENERATED IMAGES
#

# ALL DATA SAMPLE
datasetmodule = PostdamCarsDataModule(potsdam_dir,
                                      data_dir2=munit_dir1,
                                      transform=transform,
                                      batch_size=30)
datasetmodule.setup()
dataloader = datasetmodule.train_dataloader()

real, refined = next(iter(dataloader))
grid = torchvision.utils.make_grid(
    refined, normalize=True, nrow=10, pad_value=1)
torchvision.utils.save_image(grid, "figs/munit_1_sample.png")

# LIMITED DATA SAMPLE
datasetmodule = PostdamCarsDataModule(potsdam_dir,
                                      data_dir2=munit_dir2,
                                      transform=transform,
                                      batch_size=30)
datasetmodule.setup()
dataloader = datasetmodule.train_dataloader()

real, refined = next(iter(dataloader))
grid = torchvision.utils.make_grid(
    refined, normalize=True, nrow=10, pad_value=1)
torchvision.utils.save_image(grid, "figs/munit_2_sample.png")
