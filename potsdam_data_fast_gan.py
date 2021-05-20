import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

# input params
cars = np.load("../potsdam_data/potsdam_cars/cars.npy", allow_pickle=True)
cars_path = "../potsdam_data/fast_gan_cars"
r, c = 128, 256

r_center = r // 2
c_center = c // 2
cars_resized = []
for car in cars:
    car_resized = cv2.resize(car, (c, r))
    empty = np.zeros([c, c, 3], dtype=np.uint8)

    empty[c_center - r_center:c_center + r_center, :, :] = car_resized
    cars_resized.append(empty)

# fig = plt.figure(figsize=(10, 10))
# columns = 7
# rows = 7
# for i in range(1, columns*rows + 1):
#     img = cars_resized[i-1]
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)
#     plt.xticks([])
#     plt.yticks([])
# plt.tight_layout()
# plt.show()

if not os.path.isdir(cars_path):
    os.makedirs(cars_path)

for i, car in enumerate(cars_resized):
    r, c, _ = car.shape
    if r == 0 or c == 0:
        continue
    car_path = os.path.join(cars_path, "car_%d.png" % i)
    cv2.imwrite(car_path, car)
