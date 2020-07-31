import cv2
from matplotlib import pyplot as plt
import numpy as np

import os
import os.path as op

from PIL import Image


def my_PAF_release(centerA, centerB, theta=1, height=256, width=192):
    vector = np.zeros((height, width, 2))
    ax, ay = centerA[0], centerA[1]
    bx, by = centerB[0], centerB[1]

    limb_vec = centerA - centerB
    norm = np.linalg.norm(limb_vec)
    limb_vec /= norm
    bax, bay = limb_vec

    min_w = max(int(round(min(ax, bx) - theta)), 0)
    max_w = min(int(round(max(ax, bx) + theta)), width)
    min_h = max(int(round(min(ay, by) - theta)), 0)
    max_h = min(int(round(max(ay, by) + theta)), height)

    coor_w, coor_h = np.meshgrid(range(min_w, max_w), range(min_h, max_h))
    diff_x = coor_w - ax
    diff_y = coor_h - ay
    # why ? magic ?
    mask = np.abs(diff_x * bay - diff_y * bax) <= theta

    vector[coor_h, coor_w, :] = np.array([bax, bay])
    vector[coor_h, coor_w, :] = vector[coor_h, coor_w, :] * mask.astype(np.float)[:, :, np.newaxis]

    return vector


def vis_paf(pafs, img=None, height=256, width=192, centers=None, center_lab=None, limb_pairs=None, dpi=15,
            cm_paf=None, cm_center=None, paf_sample_rate=10, r=2.5):

    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi))
    if img:
        ax.imshow(img)

    limbs = pafs.shape[-1] // 2
    assert not ((centers is None) ^ (limb_pairs is None))
    if limb_pairs:
        assert limbs == len(limb_pairs)

    center_color = lambda c: cm_center[c] if cm_center else 'r'
    quiver_color = lambda l: cm_paf[l] if cm_paf else None

    for i in range(limbs):
        if centers is not None:
            c1, c2 = limb_pairs[i]
            c1_vis, c2_vis = centers[c1, 2], centers[c2, 2]
            if c1_vis != 0 and c2_vis != 0:
                c1_coor, c2_coor = centers[c1, :2], centers[c2, :2]

                # change color here
                circle_1 = plt.Circle(c1_coor, color=center_color(c1), radius=r)
                circle_2 = plt.Circle(c2_coor, color=center_color(c2), radius=r)
                ax.add_patch(circle_1)
                ax.add_patch(circle_2)

        # draw paf
        paf = pafs[:, :, 2*i: 2*i+2]
        hh, ww = np.nonzero(paf[..., 0])
        hh, ww = hh[::paf_sample_rate], ww[::paf_sample_rate]
        Q = ax.quiver(
            ww,
            hh,
            paf[hh, ww, 0],
            -paf[hh, ww, 1],
            width=0.002,
            scale=50,
            color=quiver_color(i)
        )
    if center_lab and centers is not None:
        text_base = '{}: ({:.2f}, {:.2f})'
        for lab, c in zip(center_lab, centers):
            if c[-1] == 0:
                continue

            ax.text(*c[:2], text_base.format(lab, c[0], c[1]), fontsize=15, color='w')

    plt.show()
