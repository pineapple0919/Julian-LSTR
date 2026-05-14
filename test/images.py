import os
import torch
import cv2
import json
import time
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from torch import nn
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from config import system_configs

from utils import crop_image, normalize_

from sample.vis import *


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_curves = outputs['pred_logits'], outputs['pred_curves']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        labels[labels != 1] = 0
        results = torch.cat([labels.unsqueeze(-1).float(), out_curves], dim=-1)

        return results

def kp_detection(db, nnet, image_root, debug=False, evaluator=None):
    input_size  = db.configs["input_size"]  # [h w]
    image_dir = os.path.join(image_root, "images")
    result_dir = os.path.join(image_root, "detections")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    image_names = os.listdir(image_dir)
    num_images = len(image_names)

    postprocessors = {'bbox': PostProcess()}

    for ind in tqdm(range(0, num_images), ncols=67, desc="locating kps"):
        image_file    = os.path.join(image_dir, image_names[ind])
        image         = cv2.imread(image_file)
        height, width = image.shape[0:2]

        images = np.zeros((1, 3, input_size[0], input_size[1]), dtype=np.float32)
        masks = np.ones((1, 1, input_size[0], input_size[1]), dtype=np.float32)
        orig_target_sizes = torch.tensor(input_size).unsqueeze(0).cuda()
        pad_image     = image.copy()
        pad_mask      = np.zeros((height, width, 1), dtype=np.float32)
        resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
        resized_mask  = cv2.resize(pad_mask, (input_size[1], input_size[0]))
        masks[0][0]   = resized_mask.squeeze()
        resized_image = resized_image / 255.
        normalize_(resized_image, db.mean, db.std)
        resized_image = resized_image.transpose(2, 0, 1)
        images[0]     = resized_image
        images        = torch.from_numpy(images).cuda(non_blocking=True)
        masks         = torch.from_numpy(masks).cuda(non_blocking=True)
        torch.cuda.synchronize(0)  # 0 is the GPU id
        t0            = time.time()
        outputs, weights      = nnet.test([images, masks])
        torch.cuda.synchronize(0)  # 0 is the GPU id
        t             = time.time() - t0
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if evaluator is not None:
            evaluator.add_prediction(ind, results.cpu().numpy(), t)


        if debug:
            pred = results[0].cpu().numpy()
            img  = pad_image
            img_h, img_w, _ = img.shape
            
            # 1. 篩選出模型預測有效的車道線 (預設已按信心度排序)
            valid_lanes = pred[pred[:, 0].astype(int) == 1]
            
            overlay = img.copy()
            RED = (0, 0, 255)
            txt_lines = []

            # ==========================================
            # 核心優化：數值法動態尋找消失點 (Vanishing Point)
            # ==========================================
            # 定義全域 y 測試區間 (-1 代表影像最遠端/天空，1 代表車頭底部)
            ys_search = np.linspace(-1.0, 1.0, num=500)
            dynamic_upper_y = -1.0  # 預設最高點 (若找不到交叉點的防呆機制)

            if len(valid_lanes) >= 2:
                # 取信心度最高的兩條線 (通常是左右自車道) 的 6 個多項式參數 [k, f, m, n, b, p]
                # 注意：valid_lanes 的結構為 [label, lower, upper, k, f, m, n, b, p]
                params1 = valid_lanes[0, 3:] 
                params2 = valid_lanes[1, 3:]
                
                # 分別計算這兩條線在 y_search 區間上的 x 座標
                x1 = params1[0] / (ys_search - params1[1]) ** 2 + params1[2] / (ys_search - params1[1]) + params1[3] + params1[4] * ys_search - params1[5]
                x2 = params2[0] / (ys_search - params2[1]) ** 2 + params2[2] / (ys_search - params2[1]) + params2[3] + params2[4] * ys_search - params2[5]
                
                # 尋找 x 座標差距最小的地方作為「交叉點」
                diff = np.abs(x1 - x2)
                vp_idx = np.argmin(diff)
                
                # 將交叉點稍微往下移一點點 (例如加 0.05)，避免兩條線在頂端完全黏在一起變糊
                dynamic_upper_y = ys_search[vp_idx] + 0.05 
                
                # 防呆機制：如果交叉點算出來太不合理（例如跑到車頭前面），設回合理值
                if dynamic_upper_y > 0.5:
                    dynamic_upper_y = -0.5

            # ==========================================
            # 繪圖迴圈：使用動態消失點作為統一邊界
            # ==========================================
            for i, lane in enumerate(valid_lanes):
                params = lane[3:]  # 取出 6 個多項式參數
                
                # 從動態算出的消失點 (dynamic_upper_y) 畫到車頭底部 (1.0)
                ys = np.linspace(dynamic_upper_y, 1.0, num=100)
                points = np.zeros((len(ys), 2), dtype=np.int32)

                # 像素映射：將 [-1, 1] 映射到 [0, img_h] (整張圖片)
                ys_for_pixel = (ys + 1.0) / 2.0
                points[:, 1] = (ys_for_pixel * img_h).astype(int)

                # 計算 x 座標
                points[:, 0] = ((params[0] / (ys - params[1]) ** 2 + 
                                 params[2] / (ys - params[1]) + 
                                 params[3] + params[4] * ys - 
                                 params[5]) * img_w).astype(int)

                # 畫紅色車道線
                for current_point, next_point in zip(points[:-1], points[1:]):
                    cv2.line(overlay, tuple(current_point), tuple(next_point), color=RED, thickness=15)

                if len(points) > 0:
                    cv2.putText(img, str(i), tuple(points[0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=RED, thickness=3)
                
                txt_line = " ".join([f"{p[0]},{p[1]}" for p in points])
                txt_lines.append(txt_line)

            # 影像混合與儲存
            w = 0.6
            img = ((1. - w) * img + w * overlay).astype(np.uint8)
            cv2.imwrite(os.path.join(result_dir, image_names[ind][:-4] + '.jpg'), img)
            
            # 輸出 txt (維持你原本的邏輯)
            txt_output_dir = os.path.join(image_root, "detections")
            os.makedirs(txt_output_dir, exist_ok=True)
            txt_file_path = os.path.join(txt_output_dir, image_names[ind][:-4] + ".txt")
            with open(txt_file_path, "w") as f:
                f.write("\n".join(txt_lines))

    return 0

def testing(db, nnet, image_root, debug=False, evaluator=None):
    return globals()[system_configs.sampling_function](db, nnet, image_root, debug=debug, evaluator=evaluator)