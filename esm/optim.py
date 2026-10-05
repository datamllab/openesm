"""The Muon+AdamW optimizer and linear warmdown schedule used by ESM.

The trainer uses Lightning/DDP for distributed gradient synchronization, so the
optimizer itself only needs the single-process MuonAdamW implementation.

Addapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import torch
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step_t: Tensor,
    lr_t: Tensor,
    beta1_t: Tensor,
    beta2_t: Tensor,
    eps_t: Tensor,
    wd_t: Tensor,
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    """

    p.mul_(1 - lr_t * wd_t)

    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)

    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t

    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    red_dim: int,
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """

    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(
        v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2
    )
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version.

    AdamW - Fused AdamW optimizer step.

    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})

        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        """
        AdamW update for each param in the group individually.
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1

            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])

            adamw_step_fused(
                p,
                grad,
                exp_avg,
                exp_avg_sq,
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        """
        Muon update for all params in the group (stacked for efficiency).
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        params: list[Tensor] = group["params"]
        if not params:
            return

        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(
                num_params, *shape, dtype=dtype, device=device
            )
        momentum_buffer = state["momentum_buffer"]

        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_params, shape[-2], 1)
                if shape[-2] >= shape[-1]
                else (num_params, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(
                state_shape, dtype=dtype, device=device
            )
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        muon_step_fused(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )

        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")


class WarmUpLinearWarmdownLR(_LRScheduler):
    """
    Linear warmup, constant, and warmdown learning-rate schedule.

    Warmup increases the rate to ``peak_lr``, and warmdown decreases it to
    ``final_lr_frac * peak_lr``.

    The default ESM recipe uses no warmup, a 50% warmdown phase, and a final
    learning-rate fraction of zero.
    """

    def __init__(
        self,
        optimizer,
        warmup_ratio,
        warmdown_ratio,
        final_lr_frac,
        total_steps,
        warm_up_finished_func=None,
        enable_wd_decay=False,
        resume_warmup_steps=0,
    ):
        self.warmup_steps = int(warmup_ratio * total_steps)
        self.warmdown_steps = int(warmdown_ratio * total_steps)
        self.constant_steps = total_steps - self.warmup_steps - self.warmdown_steps
        self.final_lr_frac = final_lr_frac
        self.total_steps = total_steps
        self.highest_lr = [group["lr"] for group in optimizer.param_groups]
        self.last_step = 0
        self.finished_warming_up = False
        self.warm_up_finished_func = warm_up_finished_func

        self.resume_warmup_steps = resume_warmup_steps
        self.resume_start_step = None
        self.resume_base_lr = None

        self.enable_wd_decay = enable_wd_decay
        self.initial_weight_decays = [
            group.get("weight_decay", 0) for group in optimizer.param_groups
        ]

        self._last_lr = (
            [0.0 for _ in self.highest_lr]
            if self.warmup_steps > 0
            else self.highest_lr.copy()
        )

        print("[Option 3] Linear warmdown learning-rate schedule enabled:")
        print(f"  - Warmup steps: {self.warmup_steps} ({warmup_ratio * 100:.1f}%)")
        print(f"  - Constant steps: {self.constant_steps}")
        print(
            f"  - Warmdown steps: {self.warmdown_steps} ({warmdown_ratio * 100:.1f}%)"
        )
        print(f"  - Final LR fraction: {final_lr_frac}")

        super(WarmUpLinearWarmdownLR, self).__init__(optimizer)

    def step(self):
        self.last_step += 1
        super().step()

    def _compute_schedule_lr(self):
        """Compute the base schedule before resume warmup adjustments."""

        if self.enable_wd_decay and self.total_steps > 0:
            wd_multiplier = max(0.0, 1.0 - self.last_step / self.total_steps)
            for i, group in enumerate(self.optimizer.param_groups):
                if self.initial_weight_decays[i] > 0:
                    group["weight_decay"] = (
                        self.initial_weight_decays[i] * wd_multiplier
                    )

        step = self.last_step

        if step <= self.warmup_steps:
            progress = step / self.warmup_steps if self.warmup_steps > 0 else 1.0
            return [lr * progress for lr in self.highest_lr]
        elif step <= self.warmup_steps + self.constant_steps:
            if not self.finished_warming_up:
                self.finished_warming_up = True
                if self.warm_up_finished_func is not None:
                    self.warm_up_finished_func()
            return self.highest_lr.copy()
        else:
            if not self.finished_warming_up:
                self.finished_warming_up = True
                if self.warm_up_finished_func is not None:
                    self.warm_up_finished_func()
            warmdown_progress = (
                (step - self.warmup_steps - self.constant_steps) / self.warmdown_steps
                if self.warmdown_steps > 0
                else 1.0
            )
            warmdown_progress = min(1.0, warmdown_progress)
            return [
                lr * (1.0 - warmdown_progress * (1.0 - self.final_lr_frac))
                for lr in self.highest_lr
            ]

    def get_lr(self):
        target_lrs = self._compute_schedule_lr()

        if self.resume_start_step is not None and self.resume_warmup_steps > 0:
            steps_since_resume = self.last_step - self.resume_start_step
            if steps_since_resume < self.resume_warmup_steps:
                progress = steps_since_resume / self.resume_warmup_steps
                base_lrs = (
                    self.resume_base_lr
                    if self.resume_base_lr is not None
                    else [0.0] * len(target_lrs)
                )
                self._last_lr = [
                    base + progress * (target - base)
                    for base, target in zip(base_lrs, target_lrs)
                ]
                return self._last_lr

        self._last_lr = target_lrs
        return self._last_lr

    def get_last_lr(self):
        return self._last_lr

    def state_dict(self):
        return {
            "last_step": self.last_step,
            "last_lr": self._last_lr,
            "finished_warming_up": self.finished_warming_up,
            "resume_warmup_steps": self.resume_warmup_steps,
            "resume_start_step": self.resume_start_step,
            "resume_base_lr": self.resume_base_lr,
        }

    def load_state_dict(self, state_dict):
        self.last_step = state_dict["last_step"]
        self._last_lr = state_dict["last_lr"]
        self.finished_warming_up = state_dict["finished_warming_up"]

        if self.resume_warmup_steps > 0:
            self.resume_start_step = self.last_step
            self.resume_base_lr = list(self._last_lr)
            print(
                f"[Resume Warmup] Starting at step={self.resume_start_step}; "
                f"raising the learning rate from {self.resume_base_lr} to the schedule over "
                f"{self.resume_warmup_steps} steps"
            )
        else:
            self.resume_start_step = state_dict.get("resume_start_step", None)
            self.resume_base_lr = state_dict.get("resume_base_lr", None)
