import math
import torch
from torch.optim import Optimizer

class Adan(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8,
                 weight_decay=0.0, max_grad_norm=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0 or not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index: {betas}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        super(Adan, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adan does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)
                    state['pre_grad'] = grad.clone()

                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_diff']
                pre_grad = state['pre_grad']
                beta1, beta2, beta3 = group['betas']

                state['step'] += 1
                diff = grad - pre_grad
                update = grad + beta2 * diff

                # 更新一階、二階動量與差異動量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_diff.mul_(beta2).add_(diff, alpha=1 - beta2)
                exp_avg_sq.mul_(beta3).addcmul_(update, update, value=1 - beta3)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                bias_correction3 = 1 - beta3 ** state['step']

                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction3)).add_(group['eps'])
                
                # 權重衰減
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # 更新參數
                p.addcdiv_(exp_avg + beta2 * bias_correction1 / bias_correction2 * exp_avg_diff, denom, value=-step_size)
                pre_grad.copy_(grad)

        return loss