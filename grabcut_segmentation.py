import os
import argparse

import cv2
import numpy as np
from scipy import ndimage


def mouse_events(event, x, y, flags, param=None):
    global mousepressed, markers, curr, prev, display, delete, img, \
        set_foreground, fb_ground, set_background

    if event == cv2.EVENT_LBUTTONDOWN:
        mousepressed = not mousepressed
        if not mousepressed:
            prev = None

    elif event == cv2.EVENT_MOUSEMOVE and mousepressed:
        if delete:
            cv2.circle(markers, (x, y), 10, 0, -1)
            cv2.circle(fb_ground, (x, y), 10, 0, -1)
        else:
            if set_foreground:
                cv2.circle(fb_ground, (x, y), 10, 1, -1)
            elif set_background:
                cv2.circle(fb_ground, (x, y), 10, 2, -1)
            else:
                curr = (x, y)
                if curr != prev and curr and prev:
                    cv2.line(markers, curr, prev, 255)
                prev = curr

        display = img.copy()

        display[markers == 255] = (0, 255, 0)
        display[fb_ground == 1] = (255, 0, 0)
        display[fb_ground == 2] = (0, 0, 255)

        if delete:
            cv2.putText(display, 'Delete On', (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif set_background:
            cv2.putText(display, 'Background On', (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif set_foreground:
            cv2.putText(display, 'Foreground On', (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


parser = argparse.ArgumentParser()
parser.add_argument("folder", help="folder path")
args = parser.parse_args()

train_folder = args.folder
masks_folder = train_folder + "_mask"
os.makedirs(masks_folder, exist_ok=True)

train_images = os.listdir(train_folder)
masks_images = os.listdir(masks_folder)


for img_name in train_images:
    if img_name in masks_images:
        continue

    curr, prev = None, None
    mousepressed = False
    set_foreground, set_background = False, False
    next_image = False
    delete = False

    print('Image name: ' + img_name)
    img_path = os.path.join(train_folder, img_name)
    img_original = cv2.imread(img_path, 1)
    img = cv2.resize(img_original, (512, 224))
    # img = cv2.medianBlur(img, 5)
    r, c = img.shape[:2]

    markers = np.zeros([r, c], dtype=np.uint8)
    fb_ground = np.zeros([r, c], dtype=np.uint8)
    result = np.zeros(img.shape, dtype=np.uint8)
    display = img.copy()
    filter_ = None

    cv2.namedWindow("original")
    cv2.namedWindow('result')
    cv2.setMouseCallback('original', mouse_events)

    while(True):
        cv2.imshow('original', display)
        cv2.imshow('result', result)

        key = cv2.waitKey(20) & 0xff
        if key == 27:  # esc
            break

        elif key == 68 or key == 100:  # D / d
            # delete
            delete = not delete
            set_background = False
            set_foreground = False
            mousepressed = False
        elif key == 83 or key == 115:  # S / s
            # segment mask
            mousepressed = False
            set_background = False
            set_foreground = False
            delete = False

            # To lazy to use cv2.findContours
            floodFill = ndimage.binary_fill_holes(markers).astype(np.uint8)
            floodFill[floodFill == 0] = cv2.GC_BGD
            floodFill[floodFill == 1] = cv2.GC_PR_FGD
            floodFill[fb_ground == 1] = cv2.GC_FGD
            floodFill[fb_ground == 2] = cv2.GC_BGD

            cv2.grabCut(img, floodFill, None, None,
                        None, 1, cv2.GC_INIT_WITH_MASK)

            filter_ = np.where((floodFill == 2) | (
                floodFill == 0), 0, 1).astype(np.uint8)
            result = img * filter_[:, :, np.newaxis]
            result[filter_ == 0] = (255, 0, 0)

        elif key == 66 or key == 98:  # b / B
            # set background
            set_background = not set_background
            delete = False
            set_foreground = False
            mousepressed = False

        elif key == 70 or key == 102:  # f / F
            # set foreground
            set_foreground = not set_foreground
            delete = False
            set_background = False
            mousepressed = False

    cv2.destroyAllWindows()
    if not next_image:
        img_ = img * filter_[:, :, np.newaxis]
        img_ = cv2.resize(
            img_, (img_original.shape[1], img_original.shape[0]), interpolation=cv2.INTER_NEAREST)
        # filter_ = cv2.morphologyEx(
        #     filter_, cv2.MORPH_OPEN, np.ones([5, 5], dtype=np.uint8))
        # filter_ = cv2.morphologyEx(
        #     filter_, cv2.MORPH_CLOSE, np.ones([7, 7], dtype=np.uint8))
        cv2.imwrite(os.path.join(masks_folder, img_name), img_)
    a = input('Do you want to continue? (y/n) ')
    if a == 'n':
        break
