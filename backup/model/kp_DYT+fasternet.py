import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .position_encoding import build_position_encoding
from .transformer import build_transformer
from .detr_loss import SetCriterion
from .matcher import build_matcher
from .misc import *
from sample.vis import save_debug_images_boxes
from config import system_configs
from .FasterNet import fasternet_t0, fasternet_t1, fasternet_t2, fasternet_s, fasternet_m, fasternet_l

BN_MOMENTUM = 0.1

# ================= 新增：自定義 DynamicTanh (不依賴 timm) =================
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5):
        super().__init__()
        # normalized_shape 可能是 int (nn.Linear 後) 或 tuple/list
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = normalized_shape
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # 激活階段
        x = torch.tanh(self.alpha * x)
        
        # 縮放與偏置階段 (仿 LayerNorm 邏輯)
        # 根據輸入維度自動判斷應用方式
        if x.dim() == 4:
            # 假設輸入為 (Batch, Channel, Height, Width)
            # weight 應該作用在 Channel 維度上
            return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        elif x.dim() == 3 or x.dim() == 2:
            # 假設輸入為 (Batch, Seq_Len, Channel) 或 (Batch, Channel)
            return x * self.weight + self.bias
        else:
            return x * self.weight + self.bias

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}"

def convert_ln_to_dyt(module):
    """
    遞迴遍歷模型，將所有的 LayerNorm 替換為 DynamicTanh。
    不再依賴 timm，透過類別名稱判斷。
    """
    for name, child in module.named_children():
        # 檢查是否為標準 nn.LayerNorm 或名稱中包含 LayerNorm 的自定義層 (如 FasterNet 內部的 LayerNorm)
        if isinstance(child, nn.LayerNorm) or "LayerNorm" in child.__class__.__name__:
            # 獲取原有的形狀參數
            shape = getattr(child, 'normalized_shape', None)
            if shape is None and hasattr(child, 'weight'):
                shape = child.weight.shape # 針對一些自定義 LayerNorm 的相容處理
            
            if shape is not None:
                new_module = DynamicTanh(shape)
                setattr(module, name, new_module)
        else:
            convert_ln_to_dyt(child)
    return module
# =========================================================================

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # 將 ReLU 換成 Tanh 以配合 DynamicTanh 風格
            x = torch.tanh(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.tanh = nn.Tanh() # 修改：ReLU -> Tanh
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.tanh(out)
        return out

class kp(nn.Module):
    def __init__(self,
                 flag=False,
                 block=None,
                 layers=None,
                 res_dims=None,
                 res_strides=None,
                 attn_dim=None,
                 num_queries=None,
                 aux_loss=None,
                 pos_type=None,
                 drop_out=0.1,
                 num_heads=None,
                 dim_feedforward=None,
                 enc_layers=None,
                 dec_layers=None,
                 pre_norm=None,
                 return_intermediate=None,
                 lsp_dim=None,
                 mlp_layers=None,
                 num_cls=None,
                 norm_layer=FrozenBatchNorm2d,
                 backbone_type='fasternet_t2'
                 ):
        super(kp, self).__init__()
        self.flag = flag
        self.norm_layer = norm_layer
        
        print(f"Initializing Backbone: {backbone_type}")
        
        if backbone_type == 'fasternet_t0':
            self.backbone = fasternet_t0()
            backbone_out_dim = 320
        elif backbone_type == 'fasternet_t1':
            self.backbone = fasternet_t1()
            backbone_out_dim = 512
        elif backbone_type == 'fasternet_t2':
            self.backbone = fasternet_t2()
            backbone_out_dim = 768
        elif backbone_type == 'fasternet_s':
            self.backbone = fasternet_s()
            backbone_out_dim = 1024
        elif backbone_type == 'fasternet_m':
            self.backbone = fasternet_m()
            backbone_out_dim = 1152
        elif backbone_type == 'fasternet_l':
            self.backbone = fasternet_l()
            backbone_out_dim = 1536
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        hidden_dim = attn_dim 
        self.aux_loss = aux_loss
        self.position_embedding = build_position_encoding(hidden_dim=hidden_dim, type=pos_type)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.input_proj = nn.Conv2d(backbone_out_dim, hidden_dim, kernel_size=1)
        
        self.transformer = build_transformer(hidden_dim=hidden_dim,
                                             dropout=drop_out,
                                             nheads=num_heads,
                                             dim_feedforward=dim_feedforward,
                                             enc_layers=enc_layers,
                                             dec_layers=dec_layers,
                                             pre_norm=pre_norm,
                                             return_intermediate_dec=return_intermediate)

        self.class_embed    = nn.Linear(hidden_dim, num_cls + 1)
        self.specific_embed = MLP(hidden_dim, hidden_dim, lsp_dim - 4, mlp_layers)
        self.shared_embed   = MLP(hidden_dim, hidden_dim, 4, mlp_layers)

        # === 自動轉換：不需 timm，自動處理 Backbone 與 Transformer ===
        self.backbone = convert_ln_to_dyt(self.backbone)
        self.transformer = convert_ln_to_dyt(self.transformer)

    def _train(self, *xs, **kwargs):
        images = xs[0]
        masks  = xs[1]
        features = self.backbone(images) 
        p = features[-1]                 

        pmasks = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]
        pos    = self.position_embedding(p, pmasks)
        hs, _, weights  = self.transformer(self.input_proj(p), pmasks, self.query_embed.weight, pos)
        output_class    = self.class_embed(hs)
        output_specific = self.specific_embed(hs)
        output_shared   = self.shared_embed(hs)
        output_shared   = torch.mean(output_shared, dim=-2, keepdim=True)
        output_shared   = output_shared.repeat(1, 1, output_specific.shape[2], 1)
        output_specific = torch.cat([output_specific[:, :, :, :2], output_shared, output_specific[:, :, :, 2:]], dim=-1)
        out = {'pred_logits': output_class[-1], 'pred_curves': output_specific[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_class, output_specific)
        return out, weights

    def forward(self, *xs, **kwargs):
        return self._train(*xs, **kwargs)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_curves': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class AELoss(nn.Module):
    def __init__(self,
                 debug_path=None,
                 aux_loss=None,
                 num_classes=None,
                 dec_layers=None
                 ):
        super(AELoss, self).__init__()
        self.debug_path  = debug_path
        weight_dict = {'loss_ce': 3, 'loss_curves': 5, 'loss_lowers': 2, 'loss_uppers': 2}
        # cardinality is not used to propagate loss
        matcher = build_matcher(set_cost_class=weight_dict['loss_ce'],
                                curves_weight=weight_dict['loss_curves'],
                                lower_weight=weight_dict['loss_lowers'],
                                upper_weight=weight_dict['loss_uppers'])
        losses  = ['labels', 'curves', 'cardinality']

        if aux_loss:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(num_classes=num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=1.0,
                                      losses=losses)

    def forward(self,
                iteration,
                save,
                viz_split,
                outputs,
                targets):

        gt_cluxy = [tgt[0] for tgt in targets[1:]]
        loss_dict, indices = self.criterion(outputs, gt_cluxy)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Save detected images during training
        if save:
            which_stack = 0
            save_dir = os.path.join(self.debug_path, viz_split)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_name = 'iter_{}_layer_{}'.format(iteration % 5000, which_stack)
            save_path = os.path.join(save_dir, save_name)
            with torch.no_grad():
                gt_viz_inputs = targets[0]
                tgt_labels = [tgt[:, 0].long() for tgt in gt_cluxy]
                pred_labels = outputs['pred_logits'].detach()
                prob = F.softmax(pred_labels, -1)
                scores, pred_labels = prob.max(-1)  # 4 10

                pred_curves = outputs['pred_curves'].detach()
                pred_clua3a2a1a0 = torch.cat([scores.unsqueeze(-1), pred_curves], dim=-1)

                save_debug_images_boxes(gt_viz_inputs,
                                        tgt_curves=gt_cluxy,
                                        tgt_labels=tgt_labels,
                                        pred_curves=pred_clua3a2a1a0,
                                        pred_labels=pred_labels,
                                        prefix=save_path)

        return (losses, loss_dict_reduced, loss_dict_reduced_unscaled,
                loss_dict_reduced_scaled, loss_value)
