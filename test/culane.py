import os
import cv2
import time
import torch.nn.functional as F
import torch
from torch import nn
from tqdm import tqdm
from config import system_configs

from utils import crop_image, normalize_
import matplotlib.pyplot as plt

from sample.vis import *

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

DARK_GREEN = (115, 181, 34)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PLUM = (255, 187, 255)
PINK = (180, 105, 255)
CYAN = (255, 128, 0)
CORAL = (86, 114, 255)

CHOCOLATE = (30, 105, 210)
PEACHPUFF = (185, 218, 255)
STATEGRAY = (255, 226, 198)


GT_COLOR = [PINK, CYAN, ORANGE, YELLOW, BLUE]
PRED_COLOR = [CORAL, GREEN, DARK_GREEN, PLUM, CHOCOLATE, PEACHPUFF, STATEGRAY]

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_curves']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        labels[labels != 1] = 0
        results = torch.cat([labels.unsqueeze(-1).float(), out_bbox], dim=-1)

        return results

def kp_detection(db, nnet, result_dir, debug=False, evaluator=None, repeat=1,
                 isEncAttn=False, isDecAttn=False):
    if db.split != "train":
        db_inds = db.db_inds if debug else db.db_inds
    else:
        db_inds = db.db_inds[:100] if debug else db.db_inds
    num_images = db_inds.size
    # num_images = 19
    multi_scales = db.configs["test_scales"]
    input_size  = db.configs["input_size"]  # [h w]

    postprocessors = {'curves': PostProcess()}

    for ind in tqdm(range(0, num_images), ncols=67, desc="locating kps"):
        db_ind        = db_inds[ind]
        image_file    = db.image_file(db_ind)
        image         = cv2.imread(image_file)
        raw_img = image.copy()
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        height, width = image.shape[0:2]

        for scale in multi_scales:
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
            images[0] = resized_image
            images = torch.from_numpy(images).cuda(non_blocking=True)
            masks = torch.from_numpy(masks).cuda(non_blocking=True)

            # seeking better FPS performance
            images = images.repeat(repeat, 1, 1, 1).cuda(non_blocking=True)
            masks = masks.repeat(repeat, 1, 1, 1).cuda(non_blocking=True)

            conv_features, enc_attn_weights, dec_attn_weights = [], [], []
            if isDecAttn or isEncAttn:
                hooks = [
                    nnet.model.module.layer4[-1].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)),
                    nnet.model.module.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                        lambda self, input, output: enc_attn_weights.append(output[1])),
                    nnet.model.module.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output[1]))
                ]
            torch.cuda.synchronize(0)  # 0 is the GPU id
            t0            = time.time()
            outputs, weights = nnet.test([images, masks])
            torch.cuda.synchronize(0)  # 0 is the GPU id
            t             = time.time() - t0

            # below codes are used for drawing attention maps
            if isDecAttn or isEncAttn:
                for hook in hooks:
                    hook.remove()
                conv_features = conv_features[0]
                enc_attn_weights = enc_attn_weights[0]
                dec_attn_weights = dec_attn_weights[0]

            results = postprocessors['curves'](outputs, orig_target_sizes)

            # ==========================================
            # 新增：拓撲幾何約束與動態消失點覆蓋 (Topology-Aware Post-processing)
            # ==========================================
            pred_temp = results[0].cpu().numpy()
            valid_lanes = pred_temp[pred_temp[:, 0].astype(int) == 1]
            
            ys_search = np.linspace(-1.0, 1.0, num=500)
            safe_horizon = -0.3 # 安全地平線先驗 (依據 TuSimple/CULane 特性可微調)
            
            if len(valid_lanes) >= 2:
                params1 = valid_lanes[0, 3:] 
                params2 = valid_lanes[1, 3:]
                x1 = params1[0] / (ys_search - params1[1]) ** 2 + params1[2] / (ys_search - params1[1]) + params1[3] + params1[4] * ys_search - params1[5]
                x2 = params2[0] / (ys_search - params2[1]) ** 2 + params2[2] / (ys_search - params2[1]) + params2[3] + params2[4] * ys_search - params2[5]
                diff = np.abs(x1 - x2)
                vp_idx = np.argmin(diff)
                dynamic_upper_y = ys_search[vp_idx] + 0.05 
                if dynamic_upper_y > 0.5:
                    dynamic_upper_y = safe_horizon
            elif len(valid_lanes) == 1:
                predicted_upper = valid_lanes[0, 2] 
                dynamic_upper_y = max(predicted_upper, safe_horizon)
            else:
                dynamic_upper_y = safe_horizon

            # 【關鍵一步】：將算出的完美消失點，直接覆蓋回 results Tensor 的 upper_bound 欄位 (index 2)
            # 這樣後續無論是 Evaluator 算分數，還是畫圖，都會用這條拉長後的完美曲線！
            mask = results[0, :, 0] == 1
            results[0, mask, 2] = float(dynamic_upper_y)
            # ==========================================

            if evaluator is not None:
                evaluator.add_prediction(ind, results.cpu().numpy(), t)

        if debug:
            img_lst = image_file.split('/')
            lane_debug_dir = os.path.join(result_dir, "lane_debug")
            if not os.path.exists(lane_debug_dir):
                os.makedirs(lane_debug_dir)

            # # Draw dec attn
            if isDecAttn:
                h, w = conv_features.shape[-2:]
                keep = results[0, :, 0].cpu() == 1.
                fig, axs = plt.subplots(ncols=keep.nonzero().shape[0] + 1, nrows=2, figsize=(44, 14))
                # print(keep.nonzero().shape[0], image_file)
                # colors = COLORS * 100
                for idx, ax_i in zip(keep.nonzero(), axs.T):
                    ax = ax_i[0]
                    ax.imshow(dec_attn_weights[0, idx].view(h, w).cpu())
                    ax.axis('off')
                    ax.set_title('query id: [{}]'.format(idx))
                    ax = ax_i[1]
                    preds = db.draw_annotation(ind, pred=results[0][idx].cpu().numpy(), cls_pred=None, img=raw_img)
                    ax.imshow(preds)
                    ax.axis('off')
                fig.tight_layout()
                img_path = os.path.join(lane_debug_dir, 'decAttn_{}_{}_{}.jpg'.format(
                    img_lst[-3], img_lst[-2], os.path.basename(image_file[:-4])))
                plt.savefig(img_path)
                plt.close(fig)

            # # Draw enc attn
            if isEncAttn:
                img_dir = os.path.join(lane_debug_dir, '{}_{}_{}'.format(
                    img_lst[-3], img_lst[-2], os.path.basename(image_file[:-4])))
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                f_map = conv_features
                # print('encoder attention: {}'.format(enc_attn_weights[0].shape))
                # print('feature map: {}'.format(f_map.shape))
                shape = f_map.shape[-2:]
                image_height, image_width, _ = raw_img.shape
                sattn = enc_attn_weights[0].reshape(shape + shape).cpu()
                _, label, _ = db.__getitem__(ind)  # 4, 115
                # print(db.max_points)  # 56
                for i, lane in enumerate(label):
                    if lane[0] == 0:  # Skip invalid lanes
                        continue
                    lane = lane[3:]  # remove conf, upper and lower positions
                    xs = lane[:len(lane) // 2]
                    ys = lane[len(lane) // 2:]
                    ys = ys[xs >= 0]
                    xs = xs[xs >= 0]
                    # norm_idxs = zip(ys, xs)
                    idxs = np.stack([ys * image_height, xs * image_width], axis=-1)
                    attn_idxs = np.stack([ys * shape[0], xs * shape[1]], axis=-1)

                    for idx_o, idx, num in zip(idxs, attn_idxs, range(xs.shape[0])):
                        fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(20, 14))
                        ax_i = axs.T
                        ax = ax_i[0]
                        ax.imshow(sattn[..., int(idx[0]), int(idx[1])], cmap='cividis', interpolation='nearest')
                        ax.axis('off')
                        ax.set_title('{}'.format(idx_o.astype(int)))
                        ax = ax_i[1]
                        ax.imshow(raw_img)
                        ax.add_patch(plt.Circle((int(idx_o[1]), int(idx_o[0])), color='r', radius=16))
                        ax.axis('off')
                        fig.tight_layout()

                        img_path = os.path.join(img_dir, 'encAttn_lane{}_{}_{}.jpg'.format(
                            i, num, idx_o.astype(int)))
                        plt.savefig(img_path)
                        plt.close(fig)

            # ---------- 取代原本的 db.draw_annotation 區塊 ----------
            if not isEncAttn and not isDecAttn:
                img_h, img_w, _ = image.shape
                overlay = image.copy()
                RED = (0, 0, 255)
                
                # 直接取用我們剛剛更新過 upper 的 results
                final_preds = results[0].cpu().numpy()
                final_valid_lanes = final_preds[final_preds[:, 0].astype(int) == 1]
                
                for i, lane in enumerate(final_valid_lanes):
                    # 取出覆蓋過後的動態 upper
                    upper = lane[2] 
                    params = lane[3:]
                    
                    # 取樣從動態 upper 畫到 1.0 (車頭)
                    ys = np.linspace(upper, 1.0, num=100)
                    points = np.zeros((len(ys), 2), dtype=np.int32)

                    # 正確的 [-1, 1] -> 像素 映射
                    ys_for_pixel = (ys + 1.0) / 2.0
                    points[:, 1] = (ys_for_pixel * img_h).astype(int)

                    points[:, 0] = ((params[0] / (ys - params[1]) ** 2 + 
                                     params[2] / (ys - params[1]) + 
                                     params[3] + params[4] * ys - 
                                     params[5]) * img_w).astype(int)

                    for current_point, next_point in zip(points[:-1], points[1:]):
                        cv2.line(overlay, tuple(current_point), tuple(next_point), color=RED, thickness=8)
                
                # 混合並儲存圖片
                w = 0.6
                preds_img = ((1. - w) * image + w * overlay).astype(np.uint8)
                
                cv2.imwrite(os.path.join(lane_debug_dir, img_lst[-3] + '_'
                                         + img_lst[-2] + '_'
                                         + os.path.basename(image_file[:-4]) + '.jpg'), preds_img)
                
                
    if not debug:
        exp_name = 'culane'
        evaluator.exp_name = exp_name
        _ = evaluator.eval(label='{}'.format(os.path.basename(exp_name)))


    return 0

def testing(db, nnet, result_dir, debug=False, evaluator=None, repeat=1,
            debugEnc=False, debugDec=False):
    return globals()[system_configs.sampling_function](db, nnet, result_dir, debug=debug, evaluator=evaluator,
                                                       repeat=repeat, isEncAttn=debugEnc, isDecAttn=debugDec)