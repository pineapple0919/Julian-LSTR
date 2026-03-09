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
        
        # 展平所有 batch 的預測值
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [bs * nq, cls]
        out_bbox = outputs["pred_curves"].flatten(0, 1)            # [bs * nq, 8 或 6]

        # 展平所有 batch 的標籤
        tgt_ids = torch.cat([v[:, 0] for v in targets]).long()     # [total_gts]
        tgt_bbox = torch.cat([v[:, 1:] for v in targets])          # [total_gts, 6]

        # 1. 分類代價
        cost_class = -out_prob[:, tgt_ids]

        # 2. 幾何參數代價 (強制截斷至 6 維以對齊標籤)
        # 使用廣播計算 L1 距離: [bs*nq, total_gts, 6]
        diff = torch.abs(out_bbox[:, None, :6] - tgt_bbox[None, :, :6])
        
        # 數值保護：防止 Inf 產生導致 linear_sum_assignment 崩潰
        diff = torch.clamp(diff, min=0, max=10.0)
        
        cost_lower = diff[:, :, 0] 
        cost_upper = diff[:, :, 1] 
        cost_poly  = diff[:, :, 2:].sum(dim=-1) 

        # 3. 最終代價矩陣 C (修正 self.cost_class)
        C = self.cost_class * cost_class + \
            self.curves_weight * cost_poly + \
            self.lower_weight * cost_lower + \
            self.upper_weight * cost_upper
            
        # 將矩陣轉回 [batch, queries, total_gts] 以便按 batch 處理
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets]
        indices = []
        
        # 按 batch 拆分並進行匈牙利匹配
        for i, (c, size) in enumerate(zip(C.split(sizes, -1), sizes)):
            # c[i] 代表取該 batch 對應的 GT 部分
            indices.append(linear_sum_assignment(c[i]))
            
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(set_cost_class,
                  curves_weight, lower_weight, upper_weight):
    return HungarianMatcher(cost_class=set_cost_class,
                            curves_weight=curves_weight, lower_weight=lower_weight, upper_weight=upper_weight)