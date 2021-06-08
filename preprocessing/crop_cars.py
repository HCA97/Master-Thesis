import os
import re
import json
import argparse

import cv2
import numpy as np
# import matplotlib.pyplot as plt


def load_images_cv(path, filter_car=False):
    """Load images using opencv which is much faster than PIL"""
    ret = cv2.imread(path)

    if ret is None:
        raise RuntimeError(f"{path} is not exists.")

    ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)
    if filter_car:
        r = ret[:, :, 0] == 255
        b = ret[:, :, 1] == 255
        g = ret[:, :, 2] == 0
        ret = r * b * g

    return ret


def load_cars_bbox_potsdam(path):
    coords = []

    with open(path, "r") as f:
        data = json.load(f)

        labels = data["labels"]
        polygons = data["polygons"]

        for label, polygon in zip(labels, polygons):
            if label == "car":
                coords.append(
                    np.array(polygon["coordinates"][0][:-1], dtype=int))

    return coords


def get_bbox(coord, buffer, img_shape):

    r, c = img_shape[:2]

    x_min, x_max = max(0, coord[:, 0].min() -
                       buffer), min(r, coord[:, 0].max() + buffer)
    y_min, y_max = max(0, coord[:, 1].min() -
                       buffer), min(c, coord[:, 1].max() + buffer)

    return x_min, x_max, y_min, y_max


def get_tilt(coord):
    """Returns the orientation of the bbox."""

    # index 0 is the top left corner
    d = np.linalg.norm(coord[0] - coord, axis=1)
    idx_second_longest = d.argsort()[-2]

    # get direction vector
    vec = np.array(coord[idx_second_longest] - coord[0], dtype=float)
    if np.isclose(np.linalg.norm(vec), 0):
        raise RuntimeError
    vec /= np.linalg.norm(vec)

    # align all the car respect to x axis
    ang = np.arccos(vec.dot([1, 0])) * 180 / np.pi
    pos = np.sign(np.cross([1, 0], vec))
    ang = pos*ang

    return ang


def crop_car(img, coord, buffer=5):
    ''' 
    Rotates OpenCV image around center with angle theta (in deg)
    then crops the image according to width and height.

    Reference: 
    https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
    '''

    image = img.copy()

    # bbox
    x_min, x_max, y_min, y_max = get_bbox(coord, 5, image.shape)
    cx, cy = (x_max + x_min)//2, (y_min + y_max)//2
    angle = get_tilt(coord)

    # cv2.warpAffine expects shape in (length, height)
    shape = (image.shape[1], image.shape[0])

    orig = image[y_min:y_max, x_min:x_max, :].copy()

    matrix = cv2.getRotationMatrix2D(center=(cx, cy), angle=angle, scale=1)
    image = cv2.warpAffine(image, M=matrix, dsize=shape)

    # rotate bbox
    coord_rot = (coord.dot(matrix[:, :2].T) + matrix[:, -1]).astype(np.int64)

    # crop image and delete image
    x_min, x_max, y_min, y_max = get_bbox(coord_rot, buffer, image.shape)
    chunk = image[y_min:y_max, x_min:x_max, :].copy()

    del image

    # return subimage
    return orig, chunk


def save_cars(cars, cars_path):
    if not os.path.isdir(cars_path):
        os.makedirs(cars_path)

    np.save(os.path.join(cars_path, "cars.npy"),
            np.array(cars, dtype="object"))
    for i, car in enumerate(cars):
        r, c, _ = car.shape
        if r == 0 or c == 0:
            continue
        car_path = os.path.join(cars_path, "car_%d.png" % i)
        cv2.imwrite(car_path, car)


def return_artificial_cars(json_file, buffer=5):
    root = os.path.split(json_file)[0]

    cars_succeed = []
    cars_failed = []
    with open(json_file, "r") as f:
        annotations = json.load(f)

        for annotation in annotations:
            coords = []
            labels, polygons = annotation["annotations"]["labels"], \
                annotation["annotations"]["polygons"]

            for label, polygon in zip(labels, polygons):
                if label == "car":
                    coords.append(
                        np.array(polygon["coordinates"][0][:-1], dtype=int))

            rgb_path = os.path.join(root, annotation["image_filename"])

            print(f"::Reading RBG => {rgb_path}")

            img = load_images_cv(rgb_path)
            cars_succeed += [crop_car(img, coord, buffer)[1]
                             for coord in coords]
            print("-----"*10)

    print(
        f"::Total number of cars are {len(cars_succeed)}.")
    return cars_succeed, cars_failed


def return_potsdam_cars(data_path, buffer=5, min_height=20, area_threshold=0.9, ratio_threshold=2):
    #test1 = "top_potsdam_2_10_RGB.tif"
    #test2 = "top_potsdam_2_10_label.tif"
    #test3 = "top_potsdam_2_10_annos.json"
    exp = r"top_potsdam_([0-9]+_[0-9]+)_(RGB|label|annos).[tif|json]"

    #assert re.search(exp, test1).groups() == ("2_10", "RGB")
    #assert re.search(exp, test2).groups() == ("2_10", "label")
    #assert re.search(exp, test3).groups() == ("2_10", "annos")

    data_dict = {}
    for f in os.listdir(data_path):
        if os.path.splitext(f)[-1] in [".tif", ".json"]:
            m = re.search(exp, f)
            if m:
                tile_id, image_type = m.groups()
                res = data_dict.get(tile_id, [None, None, None])
                if image_type == "label":
                    res[1] = os.path.join(data_path, f)
                elif image_type == "RGB":
                    res[0] = os.path.join(data_path, f)
                else:
                    res[2] = os.path.join(data_path, f)
                data_dict[tile_id] = res

    cars_succeed = []
    cars_failed = []
    for key in data_dict:
        rgb_path, label_path, annot_path = data_dict[key]
        print(f"::Reading RBG => {rgb_path}")
        print(f"::Reading Label => {label_path}")
        print(f"::Reading Annotation => {annot_path}")

        mask = load_images_cv(label_path, True)
        img = load_images_cv(rgb_path, False)

        coords = load_cars_bbox_potsdam(annot_path)
        print(f"::Total number of unique cars => {len(coords)}")

        for coord in coords:
            try:
                car_original, car_aligned = crop_car(img, coord, buffer)

                x_min, x_max, y_min, y_max = get_bbox(
                    coord, 5, mask.shape)
                car_mask = mask[y_min:y_max, x_min:x_max]
                r, c, _ = car_aligned.shape

                cnts, _ = cv2.findContours(car_mask.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                idx = np.argmax([len(cnt) for cnt in cnts])
                peri = cv2.arcLength(cnts[idx], True)
                approx = cv2.approxPolyDP(cnts[idx], 0.04 * peri, True)

                # plt.figure()
                # plt.imshow(car_original)
                # # plt.show()

                # plt.figure()
                # plt.imshow(car_aligned)

                # plt.figure()
                # plt.imshow(car_mask)

                # plt.show()
                # background_area = r*c
                # print(car_area, background_area)
                if r < min_height:
                    raise RuntimeError(f"Image height is short ({r}))")
                elif r == 0 or c == 0:
                    raise RuntimeError(
                        f"Image has empty rows ({r}) or cols ({c})")
                elif len(approx) < 3 or len(approx) > 5:
                    raise RuntimeError("Not a square")
                # elif car_area / background_area < area_threshold:
                #     raise RuntimeError(
                #         f"Mask of car is not complete, ratio of car and background is {car_area / (r*c)}")
                elif c < ratio_threshold*r:
                    raise RuntimeError(
                        f"Aspect ratio of car is not correct rows ({r}) and cols ({c})")
                cars_succeed.append(car_aligned)
            except RuntimeError:
                cars_failed.append(car_aligned)
        print("-----"*10)
        # break

    print(
        f"::{len(cars_succeed)} cars are succeed and {len(cars_failed)} cars are failed.")
    return cars_succeed, cars_failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Crops cars from potsdam dataset")
    parser.add_argument("--crop_artificial_cars", action="store_true",
                        dest="crop_artificial_cars", help="compute artificial cars")
    parser.add_argument(
        "--save_path", default="../potsdam_data/potsdam_cars", help="where to save")
    parser.add_argument(
        "--data_path", default="../potsdam_data/training", help="Potsdam data path")
    parser = parser.set_defaults(crop_artificial_cars=False)
    args = parser.parse_args()

    data_path = args.data_path
    save_path = args.save_path
    crop_artificial_cars = args.crop_artificial_cars

    # params
    min_height = 35  # pixles => 1.75 [m]
    ratio_threshold = 1.8

    potsdam_cars_succeed, potsdam_cars_failed = return_potsdam_cars(
        data_path, min_height=min_height, ratio_threshold=ratio_threshold)

    # succeed cars
    if potsdam_cars_succeed:
        cars_path = save_path
        save_cars(potsdam_cars_succeed, cars_path)

    # failed cars
    if potsdam_cars_failed:
        cars_path = save_path + "_failed"
        save_cars(potsdam_cars_failed, cars_path)

    # artificial cars
    if crop_artificial_cars:
        json_path = "../potsdam_data/cem-v0/vehicles_600_carssmalltrucks_outside_bnr10-bnf-defo0.05/annotations.json"
        artificial_cars_succeed, artificial_cars_failed = return_artificial_cars(
            json_path, buffer=5)

        # succeed cars
        if artificial_cars_succeed:
            cars_path = "../potsdam_data/artificial_cars"
            save_cars(artificial_cars_succeed, cars_path)

        # failed cars
        if artificial_cars_failed:
            cars_path = "../potsdam_data/artificial_failed"
            save_cars(artificial_cars_failed, cars_path)
