import os
import cv2
import numpy as np
import tqdm
import json
import argparse
import math


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="The root of the dataset")
    return parser


def parse_float(s):
    try:
        return float(s)
    except Exception:
        return float("nan")


if __name__ == "__main__":
    args = get_args().parse_args()
    culane_root = args.root
    train_list = os.path.join(culane_root, "list/train_gt.txt")
    with open(train_list, "r") as fp:
        res = fp.readlines()

    cache_dict = {}
    for line in tqdm.tqdm(res):
        info = line.split(" ")

        label_path = os.path.join(culane_root, info[1][1:])
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label_img is None:
            raise FileNotFoundError(f"Failed to read label image: {label_path}")
        H, W = label_img.shape[:2]

        txt_path = info[0][1:].replace("jpg", "lines.txt")
        txt_path = os.path.join(culane_root, txt_path)
        lanes = open(txt_path, "r").readlines()

        # (num_lanes, 60 anchors, (x,y))
        all_points = np.zeros((2, 60, 2), dtype=float)
        the_anno_row_anchor = np.arange(0, 600, 10)

        all_points[:, :, 1] = np.tile(the_anno_row_anchor, (2, 1))
        all_points[:, :, 0] = -99999  # init using no lane

        for lane_idx, lane in enumerate(lanes):
            ll = lane.strip().split()
            if len(ll) < 2:
                continue

            # split into x/y lists
            point_x = [parse_float(v) for v in ll[::2]]
            point_y = [parse_float(v) for v in ll[1::2]]

            # --- determine lane_order using the FIRST VALID POINT (not midpoint) ---
            lane_order = 0
            found_order = False
            for i0 in range(len(point_x)):
                x0f, y0f = point_x[i0], point_y[i0]
                if not (math.isfinite(x0f) and math.isfinite(y0f)):
                    continue
                x0 = int(round(x0f))
                y0 = int(round(y0f))
                # bounds check (OpenCV expects 0<=x<W, 0<=y<H)
                if 0 <= x0 < W and 0 <= y0 < H:
                    lo = int(label_img[y0, x0])
                    if lo > 0:
                        lane_order = lo
                        found_order = True
                        break
            if not found_order:
                # If nothing valid was found, skip this polyline
                # (you can log instead if you want to investigate)
                # print(f"[warn] No valid first point with label >0 in {txt_path}")
                continue

            # --- fill anchor-aligned points for this lane ---
            for i in range(len(point_x)):
                x_f = point_x[i]
                y_f = point_y[i]
                if not (math.isfinite(x_f) and math.isfinite(y_f)):
                    continue

                # map y (pixels) to row anchor index [0..59] using rounding to nearest 10
                pos = int(round(y_f / 10.0))
                if 0 <= pos < all_points.shape[1]:
                    try:
                        all_points[lane_order - 1, pos, 0] = x_f
                    except IndexError:
                        # lane_order outside [1..num_lanes]
                        print(
                            f"IndexError: index {lane_order - 1} is out of bounds for axis 0 with size {all_points.shape[0]}"
                        )
                        print(f"These are the points: {point_x}, {point_y}")
                        print(
                            f"First valid used for lane order -> x:{x_f} y:{y_f}, lane_order {lane_order}"
                        )
                        print(
                            f"Label path: {label_path}, img path: {info[0][1:]}, txt path: {txt_path}"
                        )
                        raise

        cache_dict[info[0][1:]] = all_points.tolist()

    with open(os.path.join(culane_root, "culane_anno_cache.json"), "w") as f:
        json.dump(cache_dict, f)
