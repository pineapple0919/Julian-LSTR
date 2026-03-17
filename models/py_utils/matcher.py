# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network"""

    def __init__(self, cost_class: float = 1,
                 curves_weight: float = 1, lower_weight: float = 1, upper_weight: float = 1):
        super().__init__()
        self.cost_class = cost_class # 這裡儲存的名字是 cost_class
        self.curves_weight = curves_weight
        self.lower_weight = lower_weight
        self.upper_weight = upper_weight

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # 1. 取得預測值
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [bs * nq, cls]
        out_bbox = outputs["pred_curves"].flatten(0, 1)              # [bs * nq, 6] (lower, upper, a, b, c, d)

        # 2. 解析 targets (真實標籤)
        # 此時的 targets 結構為: [class, lower, upper, x1..xn, y1..yn]
        tgt_ids = torch.cat([v[:, 0] for v in targets]).long()       # [total_gts]
        tgt_lower = torch.cat([v[:, 1] for v in targets])            # [total_gts]
        tgt_upper = torch.cat([v[:, 2] for v in targets])            # [total_gts]
        
        # 動態計算點的數量 (扣除前 3 個屬性後，剩下的是 xs 和 ys，各佔一半)
        num_points = (targets[0].shape[1] - 3) // 2
        tgt_xs = torch.cat([v[:, 3:3+num_points] for v in targets])  # [total_gts, num_points]
        tgt_ys = torch.cat([v[:, 3+num_points:] for v in targets])   # [total_gts, num_points]

        # 3. 分類代價
        cost_class = -out_prob[:, tgt_ids]

        # 4. 幾何邊界代價 (lower, upper L1 Distance)
        cost_lower = torch.abs(out_bbox[:, None, 0] - tgt_lower[None, :]) # [bs*nq, total_gts]
        cost_upper = torch.abs(out_bbox[:, None, 1] - tgt_upper[None, :]) # [bs*nq, total_gts]

        # 5. 曲線點對點代價 (Point-to-Point L1 Distance)
        # 提取預測的 a, b, c, d 係數，並擴展維度以便進行廣播運算
        a = out_bbox[:, None, 2:3] # [bs*nq, 1, 1]
        b = out_bbox[:, None, 3:4]
        c = out_bbox[:, None, 4:5]
        d = out_bbox[:, None, 5:6]
        
        # 將 GT 的 ys 擴展維度
        ys = tgt_ys[None, :, :] # [1, total_gts, num_points]
        
        # 👑 核心魔法：使用三次多項式算出預測的 X 座標
        pred_xs = a * (ys ** 3) + b * (ys ** 2) + c * ys + d  # [bs*nq, total_gts, num_points]
        
        gt_xs = tgt_xs[None, :, :] # [1, total_gts, num_points]
        valid_mask = (gt_xs >= 0)  # 過濾掉無效點 (-1e5)
        
        # 計算 X 座標的絕對誤差，並只加總有效點
        diff = torch.abs(pred_xs - gt_xs)
        cost_poly = (diff * valid_mask).sum(dim=-1) / (valid_mask.sum(dim=-1) + 1e-6) # 平均誤差

        # 6. 最終代價矩陣 C
        C = self.cost_class * cost_class + \
            self.curves_weight * cost_poly + \
            self.lower_weight * cost_lower + \
            self.upper_weight * cost_upper
            
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets]
        indices = []
        
        for i, (c, size) in enumerate(zip(C.split(sizes, -1), sizes)):
            indices.append(linear_sum_assignment(c[i]))
            
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(set_cost_class,
                  curves_weight, lower_weight, upper_weight):
    return HungarianMatcher(cost_class=set_cost_class,
                            curves_weight=curves_weight, lower_weight=lower_weight, upper_weight=upper_weight)