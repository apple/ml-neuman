#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
Save video frames to a folder
'''

import argparse
import os

import numpy as np
import cv2
from PIL import Image
import imageio
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, type=str, help='the path to the source video')
    parser.add_argument('--save_to', required=True, type=str, help='save frames to')
    parser.add_argument('--height', required=False, type=int, default=-1, help='target frame height')
    parser.add_argument('--width', required=False, type=int, default=-1, help='target frame width')
    parser.add_argument('--every_k', required=False, type=int, default=1, help='save if frame_id %% every_k == 0')
    parser.add_argument('--skip_frames', required=False, type=int, default=0, help='skip first k frames')

    opt = parser.parse_args()
    assert ((opt.height > 0) and (opt.width > 0)) or ((opt.height == -1) and (opt.width == -1))

    assert os.path.isfile(opt.video), f'{opt.video} is not a file'
    if not os.path.exists(opt.save_to):
        os.makedirs(opt.save_to)

    video_cap = cv2.VideoCapture(opt.video)
    length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    for i in tqdm(range(length), desc='Processing Frames'):
        success, image = video_cap.read()
        if not success:
            break
        if i < opt.skip_frames:
            continue
        if i % opt.every_k != 0:
            continue
        pil_image = Image.fromarray(np.uint8(image[..., ::-1]))
        if (opt.height > 0) and (opt.width > 0):
            pil_image = pil_image.resize([opt.width, opt.height], resample=Image.BILINEAR)
        image = np.array(pil_image)
        imageio.imsave(os.path.join(opt.save_to, f'{str(saved).zfill(5)}.png'), image)
        saved += 1

    print(f'In total {saved} frames have been saved to {opt.save_to}')


if __name__ == "__main__":
    main()
