import torch
import gc
import os

from esm.dataset_sft import generate_sft_dataloader
from esm.modeling_esm import ESM_NLP
from esm.optim import (
    WarmUpLinearWarmdownLR,
)
from esm.tokenizer import get_tokenizer, get_token_bytes


try:
    from lightning.pytorch import LightningModule
except ImportError:
    from pytorch_lightning import LightningModule
from esm.dataset import generate_dataloader


def _sft_trainer_debug(message):
    if os.environ.get("ESM_SFT_DEBUG", "0").lower() not in ("1", "true", "yes"):
        return
    rank = os.environ.get("RANK", "?")
    debug_ranks = os.environ.get("ESM_SFT_DEBUG_RANKS", "0").strip().lower()
    if debug_ranks not in ("all", "*"):
        enabled_ranks = {
            item.strip() for item in debug_ranks.split(",") if item.strip()
        }
        if rank not in enabled_ranks:
            return
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(f"[ESM SFT Trainer][rank={rank} local={local_rank}] {message}", flush=True)


class ModelTrainer(LightningModule):
    def __init__(self, hparams, trained_model=None):
        super().__init__()
        if isinstance(hparams, dict):
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        self._train_step_start_time = None
        self._train_start_time = None

        self._dataloader_resume_state = None

        self.full_ds = None

        tokenizer_dir = getattr(self.hparams, "tokenizer_dir", None)
        self.hparams.tokenizer_obj = tokenizer = get_tokenizer(
            tokenizer_dir=tokenizer_dir
        )

        if not hasattr(self.hparams, "tokenizer_path"):
            self.hparams.tokenizer_path = self.hparams.tokenizer

        try:
            self.token_bytes = get_token_bytes(
                device="cpu", tokenizer_dir=tokenizer_dir
            )
            print(f"  Token bytes loaded: shape={self.token_bytes.shape}")
        except Exception as e:
            print(f"  Warning: Could not load token_bytes: {e}")
            print("  BPB metrics will not be available")
            self.token_bytes = None

        print("=" * 80)
        print("TOKENIZER INFO:")
        print("  Actual tokenizer used: ESM custom BPE tokenizer")
        print("  Tokenizer location: $ESM_BASE_DIR/tokenizer/")
        print(f"  Vocab size: {tokenizer.get_vocab_size()}")
        print("=" * 80)

        if trained_model is not None:
            self.model = trained_model
        else:
            self.model = ESM_NLP(self.hparams)

        if self.hparams.compile_model:
            compile_mode = getattr(self.hparams, "compile_mode", "transformer_only")
            compile_backend = getattr(self.hparams, "compile_backend", "inductor")
            compile_dynamic = getattr(self.hparams, "compile_dynamic", False)

            if compile_mode == "full":
                print(f"\n{'=' * 80}")
                print("[torch.compile] Compiling the full model...")
                print(
                    f"[torch.compile] mode=full | backend={compile_backend} | dynamic={compile_dynamic}"
                )
                print(
                    "[torch.compile] Warning: the ESM MCMC loop uses autograd.grad and may not compile"
                )
                print("[torch.compile] The first compilation may take 5-15 minutes...")
                print(f"{'=' * 80}\n")
                import time

                start_time = time.time()
                self.model = torch.compile(
                    self.model, backend=compile_backend, dynamic=compile_dynamic
                )
                compile_time = time.time() - start_time
                print(f"\n{'=' * 80}")
                print(f"[torch.compile] ✓ Model compilation complete ({compile_time:.1f}s)")
                print(f"{'=' * 80}\n")

            elif compile_mode == "transformer_only":
                print(
                    f"[torch.compile] Compiling only the transformer (backend={compile_backend})"
                )
                if hasattr(self.model, "transformer"):
                    self.model.transformer_eager = self.model.transformer
                    self.model.transformer = torch.compile(
                        self.model.transformer,
                        backend=compile_backend,
                        dynamic=compile_dynamic,
                    )
                    print(
                        "[torch.compile] Transformer compiled; transformer_eager retained for MCMC"
                    )
                else:
                    print("[torch.compile] Warning: model has no transformer; skipping compilation")

            elif compile_mode == "disabled":
                print("[torch.compile] Compilation disabled")

            else:
                print(
                    "[torch.compile] Skipping compilation in training mode (ESM MCMC requires create_graph=True, which AOT autograd cannot double-backpropagate)"
                )

        if self.hparams.wandb_watch:
            for name, module in self.model.named_modules():
                module.name = name

    def on_sanity_check_start(self):
        _sft_trainer_debug("sanity_check_start")

    def on_sanity_check_end(self):
        _sft_trainer_debug("sanity_check_end")

    def on_validation_start(self):
        _sft_trainer_debug("validation_start")

    def on_validation_end(self):
        _sft_trainer_debug("validation_end")

    def on_train_start(self):
        _sft_trainer_debug("train_start")

        import random

        rng = getattr(self, "_rng_resume_state", None)
        if rng is not None:
            torch.random.set_rng_state(rng["torch_cpu"])
            if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(rng["torch_cuda"])
            random.setstate(rng["python"])
            self._rng_resume_state = None
            print(f"[Exact Resume] RNG states restored for rank {self.global_rank}")

        if self.hparams.debug_unused_parameters:
            for name, param in self.model.named_parameters():
                if param.requires_grad and "image_encoder" not in name:
                    print(f"registering param - {name}")
                    param.register_hook(self.create_hook(name))
                else:
                    self.model.parameters_not_to_check.add(name)

    def create_hook(self, name):
        def hook(grad):
            self.model.used_parameters.add(name)

        return hook

    @staticmethod
    def wandb_activation_hook(run, step):
        """Weights & Biases stats logging hook (optimized)."""

        def hook(module, input, output):
            if isinstance(output, tuple):
                pass
            else:
                try:
                    data = output.detach().float()
                    run.experiment.log(
                        {
                            f"activations/{module.name}_mean": data.mean().item(),
                            f"activations/{module.name}_std": data.std().item(),
                            f"activations/{module.name}_min": data.min().item(),
                            f"activations/{module.name}_max": data.max().item(),
                        },
                        step=step,
                    )
                except RuntimeError:
                    pass

        return hook

    def training_step(self, batch, batch_idx):

        if (
            not self.hparams.no_wandb
            and self.hparams.wandb_watch
            and getattr(self.hparams, "wandb_watch_level", "parameters") == "all"
            and self.global_step % self.hparams.wandb_watch_log_freq == 0
        ):
            hook_handles = []
            hook_function = self.wandb_activation_hook(
                run=self.logger, step=self.global_step
            )
            for module in self.model.modules():
                if any(
                    param.requires_grad for param in module.parameters(recurse=False)
                ):
                    handle = module.register_forward_hook(hook_function)
                    hook_handles.append(handle)

            eval_step_dict = self.eval_step(batch, "train")
            for handle in hook_handles:
                handle.remove()

        else:
            eval_step_dict = self.eval_step(batch, "train")

        self.log_metrics(eval_step_dict, "train")
        return eval_step_dict["loss"]

    def on_after_backward(self):
        if self.hparams.log_gradients:
            total_norm = 0.0
            num_parameters = 0
            num_grads_exceeding_clip_val = 0
            total_gradients = 0
            for param in self.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm
                    num_parameters += 1

                    total_gradients += torch.numel(param.grad)
                    num_grads_exceeding_clip_val += torch.sum(
                        param.grad.abs() > self.hparams.gradient_clip_val
                    )

            assert num_parameters > 0, (
                "no gradients after backwards detected please investigate"
            )
            average_norm = (total_norm / num_parameters).detach()
            percentage_clipped = (
                (num_grads_exceeding_clip_val / total_gradients) * 100
            ).detach()

            things_to_log = {}
            things_to_log["avg_gradient_norms"] = average_norm
            things_to_log["pct_gradient_clipped"] = percentage_clipped
            self.log_metrics(things_to_log, "train")

    def on_train_batch_end(self, outputs, batch, batch_idx):

        if self.hparams.debug_unused_parameters:
            all_parameters = {name for name, _ in self.model.named_parameters()}
            unused_parameters = (
                all_parameters
                - self.model.used_parameters
                - self.model.parameters_not_to_check
            )

            print(f"number of parameters total: {len(all_parameters)}")
            print(f"number of unused_parameters: {len(unused_parameters)}")
            print(f"Unused parameters: {unused_parameters}")
            print(f"Used parameters: {self.model.used_parameters}")

        if self.hparams.manual_gc_collect_every_n_steps != -1:
            if (
                self.global_step > 0
                and self.global_step % self.hparams.manual_gc_collect_every_n_steps == 0
            ):
                print("calling GC manually")
                gc.collect()
                torch.cuda.empty_cache()

        muon_warmup_steps = getattr(self.hparams, "muon_momentum_warmup_steps", 300)
        if muon_warmup_steps > 0 and self.global_step <= muon_warmup_steps:
            if hasattr(self, "trainer") and self.trainer.optimizers:
                optimizer = self.trainer.optimizers[0]
                if hasattr(optimizer, "param_groups"):
                    target_momentum = getattr(self.hparams, "muon_momentum", 0.95)
                    base_momentum = 0.85
                    frac = min(self.global_step / muon_warmup_steps, 1.0)
                    current_momentum = (
                        1 - frac
                    ) * base_momentum + frac * target_momentum
                    for group in optimizer.param_groups:
                        if group.get("kind") == "muon":
                            group["momentum"] = current_momentum

        import time as _time

        now = _time.time()
        if self._train_step_start_time is not None:
            self._last_dt = now - self._train_step_start_time
        else:
            self._last_dt = None
        self._train_step_start_time = now
        if self._train_start_time is None:
            self._train_start_time = now

    def on_save_checkpoint(self, checkpoint):

        import torch.distributed as dist
        import random

        local_dl_state = None
        try:
            train_dl = self.trainer.train_dataloader
            if train_dl is not None:
                dataset = train_dl.dataset
                if hasattr(dataset, "get_dataloader_state"):
                    local_dl_state = dataset.get_dataloader_state()
                elif (
                    hasattr(dataset, "last_state_dict")
                    and dataset.last_state_dict is not None
                ):
                    local_dl_state = dataset.last_state_dict
        except Exception:
            pass

        local_rng_state = {
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state()
            if torch.cuda.is_available()
            else None,
            "python": random.getstate(),
        }

        if dist.is_initialized() and dist.get_world_size() > 1:
            all_dl_states = [None] * dist.get_world_size()
            dist.all_gather_object(all_dl_states, local_dl_state)
            all_rng_states = [None] * dist.get_world_size()
            dist.all_gather_object(all_rng_states, local_rng_state)
        else:
            all_dl_states = [local_dl_state]
            all_rng_states = [local_rng_state]

        checkpoint["dataloader_state_dict_by_rank"] = all_dl_states
        checkpoint["rng_states_by_rank"] = all_rng_states

        print(
            f"[Checkpoint] Saving per-rank dataloader state ({len(all_dl_states)} ranks) and RNG states"
        )

    def on_load_checkpoint(self, checkpoint):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            has_orig_mod_keys = any("_orig_mod." in k for k in state_dict)
            model_has_orig_mod = any("_orig_mod." in k for k in self.state_dict())

            if has_orig_mod_keys and not model_has_orig_mod:
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k.replace("._orig_mod.", ".")] = v
                checkpoint["state_dict"] = new_state_dict
                print(
                    f"[Checkpoint] Stripped '_orig_mod' prefix from {len(state_dict)} keys"
                )
            elif not has_orig_mod_keys and model_has_orig_mod:
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        new_state_dict["model._orig_mod." + k[len("model.") :]] = v
                    else:
                        new_state_dict[k] = v
                checkpoint["state_dict"] = new_state_dict
                print(
                    f"[Checkpoint] Added '_orig_mod' prefix to {len(state_dict)} keys"
                )

        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0

        if "dataloader_state_dict_by_rank" in checkpoint:
            states = checkpoint["dataloader_state_dict_by_rank"]
            self._dataloader_resume_state = states[rank] if rank < len(states) else None

        if "rng_states_by_rank" in checkpoint:
            rng_states = checkpoint["rng_states_by_rank"]
            self._rng_resume_state = (
                rng_states[rank] if rank < len(rng_states) else None
            )
        else:
            self._rng_resume_state = None

        if self._dataloader_resume_state:
            if "cursor" in self._dataloader_resume_state:
                print(
                    f"[Checkpoint] Rank {rank} restored SFT dataloader state: "
                    f"cursor={self._dataloader_resume_state.get('cursor')}, "
                    f"consumed={self._dataloader_resume_state.get('consumed')}, "
                    f"epoch={self._dataloader_resume_state.get('epoch')}, "
                    f"it={self._dataloader_resume_state.get('it')}, "
                    f"state_version={self._dataloader_resume_state.get('state_version', 'unknown')}"
                )
            else:
                print(
                    f"[Checkpoint] Rank {rank} restored dataloader state: "
                    f"pq_idx={self._dataloader_resume_state.get('pq_idx')}, "
                    f"rg_idx={self._dataloader_resume_state.get('rg_idx')}, "
                    f"state_version={self._dataloader_resume_state.get('state_version', 'unknown')}"
                )
        if self._rng_resume_state:
            print(
                f"[Checkpoint] Rank {rank} restored RNG state: keys={list(self._rng_resume_state.keys())}"
            )

    def on_validation_epoch_start(self):
        """Reset validation accumulators at the start of each validation epoch."""
        dataset_name = getattr(
            self.hparams,
            "base_train_dataset",
            getattr(self.hparams, "dataset_name", ""),
        )
        monitor_name = getattr(self.hparams, "checkpoint_monitor_string", "")

        self._strict_valid_ppl = (
            dataset_name == "owt" or monitor_name == "valid_perplexity"
        )
        self._val_loss_sum = None
        self._val_loss_tokens = None

        self._val_ppl_loss_sum = None
        self._val_ppl_tokens = None
        self._val_bpb_nats = 0.0
        self._val_bpb_bytes = 0
        self._val_supervised_tokens = 0
        self._val_empty_supervision_batches = 0

    def validation_step(self, batch, batch_idx):
        if batch_idx < 2:
            _sft_trainer_debug(f"validation_step_start batch_idx={batch_idx}")

        token_bytes = self.token_bytes
        if token_bytes is not None and token_bytes.device != self.device:
            token_bytes = token_bytes.to(self.device)
        eval_step_dict = self.eval_step(batch, "valid", token_bytes)
        if batch_idx < 2:
            keys = ",".join(sorted(eval_step_dict.keys()))
            _sft_trainer_debug(
                f"validation_step_eval_done batch_idx={batch_idx} keys={keys}"
            )
        if getattr(self.trainer, "sanity_checking", False):
            if batch_idx < 2:
                _sft_trainer_debug(
                    f"validation_step_sanity_skip_logging batch_idx={batch_idx}"
                )
            return

        loss_value = eval_step_dict.get("loss", None)
        supervised_tokens = eval_step_dict.get("supervised_tokens", 0)
        if loss_value is not None:
            if isinstance(loss_value, torch.Tensor):
                loss_tensor = loss_value.detach().to(
                    device=self.device, dtype=torch.float32
                )
            else:
                loss_tensor = torch.tensor(
                    loss_value, device=self.device, dtype=torch.float32
                )
            if isinstance(supervised_tokens, torch.Tensor):
                token_tensor = supervised_tokens.detach().to(
                    device=self.device, dtype=torch.float32
                )
            else:
                token_tensor = torch.tensor(
                    supervised_tokens, device=self.device, dtype=torch.float32
                )
            if self._val_loss_sum is None:
                self._val_loss_sum = torch.zeros(
                    (), device=self.device, dtype=torch.float32
                )
                self._val_loss_tokens = torch.zeros(
                    (), device=self.device, dtype=torch.float32
                )
            self._val_loss_sum = self._val_loss_sum + loss_tensor * token_tensor
            self._val_loss_tokens = self._val_loss_tokens + token_tensor

        ppl_loss_value = eval_step_dict.get("final_step_loss", None)
        if self._strict_valid_ppl and ppl_loss_value is not None:
            if isinstance(ppl_loss_value, torch.Tensor):
                ppl_loss_tensor = ppl_loss_value.detach().to(
                    device=self.device, dtype=torch.float32
                )
            else:
                ppl_loss_tensor = torch.tensor(
                    ppl_loss_value, device=self.device, dtype=torch.float32
                )
            if isinstance(supervised_tokens, torch.Tensor):
                ppl_token_tensor = supervised_tokens.detach().to(
                    device=self.device, dtype=torch.float32
                )
            else:
                ppl_token_tensor = torch.tensor(
                    supervised_tokens, device=self.device, dtype=torch.float32
                )
            if self._val_ppl_loss_sum is None:
                self._val_ppl_loss_sum = torch.zeros(
                    (), device=self.device, dtype=torch.float32
                )
                self._val_ppl_tokens = torch.zeros(
                    (), device=self.device, dtype=torch.float32
                )
            self._val_ppl_loss_sum = (
                self._val_ppl_loss_sum + ppl_loss_tensor * ppl_token_tensor
            )
            self._val_ppl_tokens = self._val_ppl_tokens + ppl_token_tensor
        self.log_metrics(eval_step_dict, "valid")

        bpb_nats = eval_step_dict.get("bpb_nats", 0)
        bpb_bytes = eval_step_dict.get("bpb_bytes", 0)
        if isinstance(bpb_nats, torch.Tensor):
            bpb_nats = bpb_nats.item()
        if isinstance(bpb_bytes, torch.Tensor):
            bpb_bytes = bpb_bytes.item()
        self._val_bpb_nats += bpb_nats
        self._val_bpb_bytes += bpb_bytes
        supervised_tokens = eval_step_dict.get("supervised_tokens", 0)
        empty_supervision_batch = eval_step_dict.get("empty_supervision_batch", 0)
        if isinstance(supervised_tokens, torch.Tensor):
            supervised_tokens = supervised_tokens.item()
        if isinstance(empty_supervision_batch, torch.Tensor):
            empty_supervision_batch = empty_supervision_batch.item()
        self._val_supervised_tokens += supervised_tokens
        self._val_empty_supervision_batches += empty_supervision_batch

        if not hasattr(self, "_last_valid_metrics"):
            self._last_valid_metrics = {}
        for k, v in eval_step_dict.items():
            if k in ("bpb_nats", "bpb_bytes"):
                continue
            if k == "loss":
                continue
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                self._last_valid_metrics[k] = v.detach().item()
            elif isinstance(v, (int, float)):
                self._last_valid_metrics[k] = v
        if batch_idx < 2:
            _sft_trainer_debug(f"validation_step_done batch_idx={batch_idx}")

    def on_validation_epoch_end(self):
        """Compute epoch-level BPB from accumulated nats/bytes and override the cached value."""
        if getattr(self.trainer, "sanity_checking", False):
            _sft_trainer_debug("validation_epoch_end_sanity_skip_logging")
            return

        import math
        import torch.distributed as dist

        loss_sum = self._val_loss_sum
        loss_tokens = self._val_loss_tokens
        if loss_sum is None:
            loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            loss_tokens = torch.zeros((), device=self.device, dtype=torch.float32)
        else:
            loss_sum = loss_sum.clone()
            loss_tokens = loss_tokens.clone()

        if dist.is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_tokens, op=dist.ReduceOp.SUM)

        if loss_tokens.item() > 0:
            epoch_loss = loss_sum / loss_tokens
        else:
            epoch_loss = torch.tensor(
                float("inf"), device=self.device, dtype=torch.float32
            )

        valid_ppl = None
        if self._strict_valid_ppl:
            ppl_loss_sum = self._val_ppl_loss_sum
            ppl_tokens = self._val_ppl_tokens
            if ppl_loss_sum is None:
                ppl_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
                ppl_tokens = torch.zeros((), device=self.device, dtype=torch.float32)
            else:
                ppl_loss_sum = ppl_loss_sum.clone()
                ppl_tokens = ppl_tokens.clone()
            if dist.is_initialized():
                dist.all_reduce(ppl_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(ppl_tokens, op=dist.ReduceOp.SUM)
            if ppl_tokens.item() > 0:
                valid_ppl = torch.exp((ppl_loss_sum / ppl_tokens).clamp(max=80.0))
            else:
                valid_ppl = torch.tensor(
                    float("inf"), device=self.device, dtype=torch.float32
                )

        if self._val_bpb_bytes > 0:
            epoch_bpb = self._val_bpb_nats / (math.log(2) * self._val_bpb_bytes)
        else:
            epoch_bpb = float("inf")

        if not hasattr(self, "_last_valid_metrics"):
            self._last_valid_metrics = {}
        self._last_valid_metrics["loss"] = epoch_loss.detach().item()
        self._last_valid_metrics["bpb"] = epoch_bpb
        if valid_ppl is not None:
            self._last_valid_metrics["perplexity"] = valid_ppl.detach().item()
        self._last_valid_metrics["supervised_tokens"] = self._val_supervised_tokens
        self._last_valid_metrics["empty_supervision_batch"] = (
            self._val_empty_supervision_batches
        )
        self.log("valid_loss", epoch_loss, prog_bar=True, sync_dist=False)
        if valid_ppl is not None:
            self.log("valid_perplexity", valid_ppl, prog_bar=True, sync_dist=False)

        if getattr(self.trainer, "is_global_zero", True):
            valid_summary = (
                f"[Validation] global_step={self.global_step} | "
                f"valid_loss: {epoch_loss.detach().item():.6f} | "
                f"valid_bpb: {epoch_bpb:.6f}"
            )
            if valid_ppl is not None:
                valid_summary += f" | valid_ppl: {valid_ppl.detach().item():.4f}"
            valid_summary += (
                f" | valid_supervised_tokens: {self._val_supervised_tokens} | "
                f"valid_empty_supervision_batch: {self._val_empty_supervision_batches}"
            )
            print(valid_summary, flush=True)

        if self.logger is not None:
            try:
                self.logger.experiment.log(
                    {
                        "valid_loss": epoch_loss.detach().item(),
                        "valid_bpb": epoch_bpb,
                        "valid_supervised_tokens": self._val_supervised_tokens,
                        "valid_empty_supervision_batch": self._val_empty_supervision_batches,
                    },
                    step=self.global_step,
                )
                if valid_ppl is not None:
                    self.logger.experiment.log(
                        {"valid_perplexity": valid_ppl.detach().item()},
                        step=self.global_step,
                    )
            except Exception:
                pass

    def eval_step(self, batch, phase, token_bytes=None):
        things_to_log = self.model.forward_loss_wrapper(
            batch, phase, token_bytes=token_bytes
        )

        return things_to_log

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        return self.configure_optimizers_nlp()

    def get_optimizer(self, optimizer_parameters):
        """Construct the paper optimizer; alternate optimizers were removed."""

        if self.hparams.optimizer != "muon_adamw":
            raise ValueError("ESM training requires optimizer='muon_adamw'.")
        from esm.optim import MuonAdamW

        class PLMuonAdamW(MuonAdamW):
            @torch.no_grad()
            def step(self, closure=None):
                if closure is not None:
                    with torch.enable_grad():
                        closure()
                super().step()

        return PLMuonAdamW(optimizer_parameters)

    def on_warm_up_finished(self):
        if hasattr(self.model, "warm_up_finished"):
            self.model.warm_up_finished()
            print("Warm up finished, calling self.model.warm_up_finished()")
        else:
            print(
                "Warm up finished, no self.model.warm_up_finished() exists so not doing anything"
            )

    def get_lr_scheduler(self, optimizer):
        enable_wd_decay = getattr(self.hparams, "dynamic_wd", False)
        if not getattr(self.hparams, "linear_warmdown", False):
            raise ValueError("ESM training requires linear_warmdown=true.")
        return WarmUpLinearWarmdownLR(
            optimizer,
            warmup_ratio=getattr(self.hparams, "warmup_ratio", 0.0),
            warmdown_ratio=getattr(self.hparams, "warmdown_ratio", 0.5),
            final_lr_frac=getattr(self.hparams, "final_lr_frac", 0.0),
            total_steps=self.hparams.max_steps,
            warm_up_finished_func=self.on_warm_up_finished,
            enable_wd_decay=enable_wd_decay,
            resume_warmup_steps=getattr(self.hparams, "resume_warmup_steps", 0),
        )

    def _split_tf_head_params(self):
        tf_head_matrix_params = []
        tf_head_scalar_params = []
        if not bool(getattr(self.hparams, "use_tf_head", False)):
            return tf_head_matrix_params, tf_head_scalar_params
        if not hasattr(self.model, "tf_head") or self.model.tf_head is None:
            return tf_head_matrix_params, tf_head_scalar_params

        for _, param in self.model.tf_head.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                tf_head_matrix_params.append(param)
            else:
                tf_head_scalar_params.append(param)
        return tf_head_matrix_params, tf_head_scalar_params

    def _validate_optimizer_param_coverage(self, optimizer):
        named_trainable_params = [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]
        id_to_name = {id(param): name for name, param in named_trainable_params}
        seen_param_groups = {}
        duplicate_params = []

        for group_idx, group in enumerate(optimizer.param_groups):
            for param in group["params"]:
                if not getattr(param, "requires_grad", False):
                    continue
                param_id = id(param)
                if param_id in seen_param_groups:
                    duplicate_params.append(
                        (
                            id_to_name.get(param_id, "<unnamed>"),
                            seen_param_groups[param_id],
                            group_idx,
                        )
                    )
                else:
                    seen_param_groups[param_id] = group_idx

        missing_params = [
            (name, param.numel())
            for name, param in named_trainable_params
            if id(param) not in seen_param_groups
        ]
        if not missing_params and not duplicate_params:
            return

        messages = ["Optimizer parameter coverage check failed."]
        if missing_params:
            preview = ", ".join(
                f"{name}({numel:,})" for name, numel in missing_params[:20]
            )
            if len(missing_params) > 20:
                preview += f", ... +{len(missing_params) - 20} more"
            messages.append(f"Missing trainable params: {preview}")
        if duplicate_params:
            preview = ", ".join(
                f"{name}(groups {first}->{second})"
                for name, first, second in duplicate_params[:20]
            )
            if len(duplicate_params) > 20:
                preview += f", ... +{len(duplicate_params) - 20} more"
            messages.append(f"Duplicate optimizer params: {preview}")
        raise RuntimeError("\n".join(messages))

    def _alpha_optimizer_params(self):
        return [self.model.alpha] if self.model.alpha.requires_grad else []

    def _configure_muon_adamw_optimizer(self):
        """
        Configure the mixed Muon and AdamW optimizer.

        Parameter groups follow the ESM paper recipe:
        - ``alpha`` uses AdamW with a high learning rate and no decay.
        - Embeddings use AdamW with a separate learning rate and no decay.
        - ``vocab_to_embed`` uses a conservative AdamW learning rate.
        - Scalar transformer and head parameters use AdamW without decay.
        - Matrix transformer and head parameters use Muon, grouped by shape.

        Parameters inside the MCMC loop use conservative rates because they
        require higher-order gradients. Lightning/DDP synchronizes gradients;
        MuonAdamW performs local parameter updates.
        """

        muon_lr = getattr(self.hparams, "muon_lr", 0.02)
        muon_momentum = getattr(self.hparams, "muon_momentum", 0.95)
        muon_ns_steps = getattr(self.hparams, "muon_ns_steps", 5)
        muon_beta2 = getattr(self.hparams, "muon_beta2", 0.95)
        adam_betas = (self.hparams.beta1, self.hparams.beta2)

        adamw_embedding_lr = float(self.hparams.adamw_embedding_lr)
        adamw_vocab_to_embed_lr = float(self.hparams.adamw_vocab_to_embed_lr)
        adamw_scalar_lr = float(self.hparams.adamw_scalar_lr)
        if min(adamw_embedding_lr, adamw_vocab_to_embed_lr, adamw_scalar_lr) <= 0:
            raise ValueError(
                "The ESM paper path requires positive absolute AdamW learning rates."
            )
        dmodel_scale = (self.hparams.embedding_dim / 768) ** -0.5
        embedding_lr = adamw_embedding_lr * dmodel_scale
        vocab_to_embed_lr = adamw_vocab_to_embed_lr * dmodel_scale
        scalar_lr = adamw_scalar_lr * dmodel_scale

        alpha_lr = (
            self.hparams.mcmc_step_size_lr_multiplier * self.hparams.peak_learning_rate
        )

        alpha_params = self._alpha_optimizer_params()
        embedding_params = list(self.model.embeddings.parameters())

        vocab_to_embed_params = []
        if (
            hasattr(self.model, "vocab_to_embed")
            and self.model.vocab_to_embed is not None
        ):
            vocab_to_embed_params = list(self.model.vocab_to_embed.parameters())

        transformer_matrix_params = []
        transformer_scalar_params = []
        for param in self.model.transformer.parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                transformer_matrix_params.append(param)
            else:
                transformer_scalar_params.append(param)
        tf_head_matrix_params, tf_head_scalar_params = self._split_tf_head_params()

        param_groups = []

        if alpha_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=alpha_params,
                    lr=alpha_lr,
                    betas=adam_betas,
                    eps=1e-10,
                    weight_decay=0.0,
                )
            )
        if embedding_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=embedding_params,
                    lr=embedding_lr,
                    betas=adam_betas,
                    eps=1e-10,
                    weight_decay=0.0,
                )
            )
        if vocab_to_embed_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=vocab_to_embed_params,
                    lr=vocab_to_embed_lr,
                    betas=adam_betas,
                    eps=1e-10,
                    weight_decay=0.0,
                )
            )
        scalar_params = transformer_scalar_params + tf_head_scalar_params
        if scalar_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=scalar_params,
                    lr=scalar_lr,
                    betas=adam_betas,
                    eps=1e-10,
                    weight_decay=0.0,
                )
            )

        shape_groups = {}
        muon_matrix_params = transformer_matrix_params + tf_head_matrix_params
        for p in muon_matrix_params:
            shape_groups.setdefault(p.shape, []).append(p)

        for shape in sorted(shape_groups.keys()):
            group_params = shape_groups[shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=muon_lr,
                    momentum=muon_momentum,
                    ns_steps=muon_ns_steps,
                    beta2=muon_beta2,
                    weight_decay=self.hparams.weight_decay,
                )
            )

        optimizer = self.get_optimizer(param_groups)
        self._validate_optimizer_param_coverage(optimizer)

        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        lr_scheduler = self.get_lr_scheduler(optimizer)

        num_tf_head_matrix_params = sum(p.numel() for p in tf_head_matrix_params)
        num_tf_head_scalar_params = sum(p.numel() for p in tf_head_scalar_params)
        num_tf_head_params = num_tf_head_matrix_params + num_tf_head_scalar_params
        num_tf_head_matrix_muon_params = sum(p.numel() for p in tf_head_matrix_params)
        num_muon_params = sum(p.numel() for p in muon_matrix_params)
        num_adamw_params = (
            sum(p.numel() for p in alpha_params)
            + sum(p.numel() for p in embedding_params)
            + sum(p.numel() for p in vocab_to_embed_params)
            + sum(p.numel() for p in transformer_scalar_params)
            + num_tf_head_scalar_params
        )
        total_optimized_params = num_muon_params + num_adamw_params
        print("=" * 80)
        print("[Muon+AdamW] Hybrid optimizer enabled:")
        print(f"  Muon groups: {len(shape_groups)} (grouped by shape)")
        print(
            f"  Muon params: {num_muon_params:,} ({num_muon_params / total_optimized_params * 100:.1f}%)"
        )
        print(
            f"  AdamW params: {num_adamw_params:,} ({num_adamw_params / total_optimized_params * 100:.1f}%)"
        )
        if num_tf_head_params > 0:
            print(
                f"  TF head params: {num_tf_head_params:,} "
                f"(matrix/Muon: {num_tf_head_matrix_muon_params:,}, "
                f"scalar/AdamW: {num_tf_head_scalar_params:,})"
            )
        print(
            f"  Muon LR: {muon_lr}, momentum: {muon_momentum}, ns_steps: {muon_ns_steps}, beta2: {muon_beta2}"
        )
        print(f"  Alpha LR: {alpha_lr} (AdamW) [ESM-specific]")
        print(f"  Embedding LR: {embedding_lr} (AdamW)")
        print(f"  vocab_to_embed LR: {vocab_to_embed_lr} (AdamW) [ESM-specific, used by MCMC]")
        print(f"  Scalar LR: {scalar_lr} (AdamW)")
        print(
            f"  dmodel scaling: {dmodel_scale:.4f} (dim={self.hparams.embedding_dim})"
        )
        for shape, params in sorted(shape_groups.items()):
            print(f"  Muon group shape={shape}: {len(params)} params")
        print("=" * 80)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def configure_optimizers_nlp(self):
        if self.hparams.model_name != "esm":
            raise NotImplementedError(
                f"ESM trainer does not support model {self.hparams.model_name!r}."
            )
        return self._configure_muon_adamw_optimizer()

    def train_dataloader(self):

        tokenizer = (
            self.hparams.tokenizer_obj
            if hasattr(self.hparams, "tokenizer_obj")
            else self.hparams.tokenizer
        )
        base_train_dataset = getattr(self.hparams, "base_train_dataset", "dclm")
        base_train_data_dir = getattr(self.hparams, "base_train_data_dir", "")
        base_train_data_dir = base_train_data_dir or None

        resume_state = getattr(self, "_dataloader_resume_state", None)
        self._dataloader_resume_state = None

        if getattr(self.hparams, "dataset_name", "esm") == "esm_sft":
            _sft_trainer_debug(
                f"train_dataloader_start batch_size={self.hparams.batch_size_per_device} "
                f"max_len={self.hparams.context_length} max_iter={self.hparams.max_steps * self.hparams.accumulate_grad_batches}"
            )
            train_dataloader = generate_sft_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.max_steps * self.hparams.accumulate_grad_batches,
                split="train",
                device=self.device,
                resume_state_dict=resume_state,
            )
            _sft_trainer_debug("train_dataloader_done")
        else:
            train_dataloader = generate_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.max_steps * self.hparams.accumulate_grad_batches,
                split="train",
                device=self.device,
                resume_state_dict=resume_state,
                base_train_dataset=base_train_dataset,
                base_train_data_dir=base_train_data_dir,
            )
        return train_dataloader

    def val_dataloader(self):

        tokenizer = (
            self.hparams.tokenizer_obj
            if hasattr(self.hparams, "tokenizer_obj")
            else self.hparams.tokenizer
        )
        base_train_dataset = getattr(self.hparams, "base_train_dataset", "dclm")
        base_train_data_dir = getattr(self.hparams, "base_train_data_dir", "")
        base_train_data_dir = base_train_data_dir or None

        if getattr(self.hparams, "dataset_name", "esm") == "esm_sft":
            _sft_trainer_debug(
                f"val_dataloader_start batch_size={self.hparams.batch_size_per_device} "
                f"max_len={self.hparams.context_length} max_iter={self.hparams.val_steps}"
            )
            val_dataloader = generate_sft_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.val_steps,
                split="val",
                device=self.device,
            )
            _sft_trainer_debug("val_dataloader_done")
        else:
            val_dataloader = generate_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.val_steps,
                split="val",
                device=self.device,
                resume_state_dict=None,
                base_train_dataset=base_train_dataset,
                base_train_data_dir=base_train_data_dir,
            )

        return val_dataloader

    def log_metrics(self, metrics_dict, phase):
        scalar_metrics = {}
        keys = list(metrics_dict.keys())
        for key in keys:
            if key in ("bpb_nats", "bpb_bytes"):
                continue
            if key == "bpb" and phase != "train":
                continue
            if key == "loss" and phase == "valid":
                continue
            if (
                key in ("supervised_tokens", "empty_supervision_batch")
                and phase != "train"
            ):
                continue

            value = metrics_dict[key]
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                self.logger.experiment.log(
                    {
                        f"{phase}_{key}_mean": value.detach().mean(),
                        f"{phase}_{key}_std": value.detach().std(),
                    }
                )

            elif isinstance(value, torch.Tensor) and value.dim() == 0:
                scalar_metrics[f"{phase}_{key}"] = value.detach()
            elif isinstance(value, (int, float)):
                scalar_metrics[f"{phase}_{key}"] = value
            else:
                raise ValueError(
                    f"unsupported type/format in log_metrics, type:, {type(value)}, key: {key}"
                )

        if scalar_metrics:
            if phase == "train":
                self.log_dict(
                    scalar_metrics,
                    sync_dist=True,
                    prog_bar=True,
                    on_step=True,
                    on_epoch=False,
                )
            else:
                self.log_dict(
                    scalar_metrics,
                    sync_dist=True,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                )

        if phase == "train" and len(self.trainer.optimizers) > 0:
            optimizer = self.trainer.optimizers[0]

            for i, group in enumerate(optimizer.param_groups):
                group_lr = group["lr"]
                group_wd = group.get("weight_decay", 0)
                self.log(
                    f"lr/param_group_{i}",
                    group_lr,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                )
                self.log(
                    f"wd/param_group_{i}",
                    group_wd,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                )

            current_lr = optimizer.param_groups[-1]["lr"]
            self.log("Global_LR", current_lr, on_step=True, on_epoch=False)

            if len(optimizer.param_groups) > 1:
                alpha_lr = optimizer.param_groups[0]["lr"]
                self.log("Alpha_LR", alpha_lr, on_step=True, on_epoch=False)

        if phase == "train" and self.hparams.mcmc_step_size_learnable:
            self.log(
                "Alpha_MCMC_Step_Size",
                self.model.alpha.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

        if phase == "train" and hasattr(self, "trainer") and self.trainer is not None:
            import time as _time

            current_step = self.global_step
            max_steps = self.hparams.max_steps
            progress_pct = 100.0 * current_step / max_steps if max_steps > 0 else 0
            self.log(
                "step", float(current_step), prog_bar=True, on_step=True, on_epoch=False
            )
            self.log(
                "progress_pct",
                progress_pct,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )

            if torch.cuda.is_available():
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
                self.log(
                    "gpu_mem_allocated_gb",
                    gpu_mem_allocated,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                )
                self.log(
                    "gpu_mem_reserved_gb",
                    gpu_mem_reserved,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                )

            if self.trainer.is_global_zero:
                dt_ms = (getattr(self, "_last_dt", None) or 0.0) * 1000.0
                wall_elapsed = 0.0
                if self._train_start_time is not None:
                    wall_elapsed = _time.time() - self._train_start_time
                total_min = wall_elapsed / 60.0

                lrm = 1.0
                if len(self.trainer.optimizers) > 0:
                    opt = self.trainer.optimizers[0]

                    cur_lr = opt.param_groups[-1]["lr"]
                    peak_lr = self.hparams.peak_learning_rate
                    lrm = cur_lr / peak_lr if peak_lr > 0 else 1.0

                num_gpus = getattr(self.hparams, "num_gpus", 1)
                tokens_per_step = (
                    num_gpus
                    * self.hparams.batch_size_per_device
                    * self.hparams.context_length
                    * self.hparams.accumulate_grad_batches
                )
                tok_per_sec = tokens_per_step / (dt_ms / 1000.0) if dt_ms > 0 else 0.0

                try:
                    num_params = sum(
                        p.numel() for p in self.model.parameters() if p.requires_grad
                    )
                    flops_per_token = 6 * num_params

                    peak_flops_per_gpu = 989e12
                    gpu_peak_flops = num_gpus * peak_flops_per_gpu
                    actual_flops_per_sec = tok_per_sec * flops_per_token
                    mfu = (
                        100.0 * actual_flops_per_sec / gpu_peak_flops
                        if gpu_peak_flops > 0
                        else 0.0
                    )
                except Exception:
                    mfu = 0.0

                epoch = self.current_epoch + 1

                if current_step > 0 and wall_elapsed > 0 and max_steps > 0:
                    steps_remaining = max_steps - current_step
                    sec_per_step = wall_elapsed / current_step
                    eta_min = steps_remaining * sec_per_step / 60.0
                    eta_str = f" | eta: {eta_min:.1f}m"
                else:
                    eta_str = ""

                loss_val = metrics_dict.get("loss", 0.0)
                if isinstance(loss_val, torch.Tensor):
                    loss_val = loss_val.item()

                last_valid = getattr(self, "_last_valid_metrics", {})
                callback_metrics = getattr(self.trainer, "callback_metrics", {})

                def _metric_to_float(metric):
                    if metric is None:
                        return None
                    if isinstance(metric, torch.Tensor):
                        return metric.detach().item()
                    if isinstance(metric, (int, float)):
                        return metric
                    return None

                valid_loss_val = _metric_to_float(
                    callback_metrics.get("valid_loss")
                    if callback_metrics is not None
                    else None
                )
                if valid_loss_val is None:
                    valid_loss_val = last_valid.get("loss", None)
                valid_bpb_val = last_valid.get("bpb", None)
                valid_ppl_val = _metric_to_float(
                    callback_metrics.get("valid_perplexity")
                    if callback_metrics is not None
                    else None
                )
                if valid_ppl_val is None:
                    valid_ppl_val = last_valid.get("perplexity", None)
                valid_str = ""
                if valid_loss_val is not None:
                    valid_str += f" | valid_loss: {valid_loss_val:.4f}"
                if valid_bpb_val is not None:
                    valid_str += f" | valid_bpb: {valid_bpb_val:.4f}"
                if valid_ppl_val is not None:
                    valid_str += f" | valid_ppl: {valid_ppl_val:.2f}"

                print(
                    f"step {current_step:05d}/{max_steps} ({progress_pct:.2f}%) | "
                    f"loss: {loss_val:.6f}"
                    f"{valid_str} | "
                    f"lrm: {lrm:.2f} | "
                    f"dt: {dt_ms:.2f}ms | "
                    f"tok/sec: {tok_per_sec:,.0f} | "
                    f"mfu: {mfu:.2f} | "
                    f"epoch: {epoch} | "
                    f"total time: {total_min:.2f}m"
                    f"{eta_str}",
                    flush=True,
                )
