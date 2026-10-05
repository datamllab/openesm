"""Checkpoint callbacks that guard against low disk space."""

import re
import shutil
from pathlib import Path

try:
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
except ImportError:
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint


class DiskAwareCheckpoint(ModelCheckpoint):
    """ModelCheckpoint with a disk-space check before saving."""

    def __init__(self, *args, min_free_gb=10, cleanup_on_low_space=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_free_gb = min_free_gb
        self.cleanup_on_low_space = cleanup_on_low_space

    def _check_disk_space(self):
        """Return the available disk space in GiB."""
        if self.dirpath:
            Path(self.dirpath).mkdir(parents=True, exist_ok=True)
            stat = shutil.disk_usage(self.dirpath)
            free_gb = stat.free / (1024**3)
            return free_gb
        return float("inf")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints while preserving ``last.ckpt``."""
        if not self.dirpath:
            return

        ckpt_dir = Path(self.dirpath)
        if not ckpt_dir.exists():
            return

        ckpts = [f for f in ckpt_dir.glob("*.ckpt") if f.name != "last.ckpt"]

        for ckpt in ckpts:
            try:
                ckpt.unlink()
                print(f"[DiskAware] Removed old checkpoint: {ckpt.name}")
            except Exception as e:
                print(f"[DiskAware] Failed to remove {ckpt.name}: {e}")

    def _save_checkpoint(self, trainer, filepath):
        """Check disk space before saving a checkpoint."""
        free_gb = self._check_disk_space()

        if free_gb < self.min_free_gb:
            message = (
                f"[DiskAware] Checkpoint saving stopped: free disk space "
                f"{free_gb:.1f}GB is below the {self.min_free_gb:.1f}GB threshold at {self.dirpath}."
            )
            if not self.cleanup_on_low_space:
                raise RuntimeError(
                    f"{message} Old checkpoints were not removed because best/top-k checkpoints must be preserved."
                )

            print(f"{message} cleanup_on_low_space=True; removing old checkpoints...")
            self._cleanup_old_checkpoints()

            free_gb = self._check_disk_space()
            print(f"[DiskAware] Free disk space after cleanup: {free_gb:.1f}GB")

        super()._save_checkpoint(trainer, filepath)

    def _temporarily_align_completed_for_save(self, trainer):
        """Align checkpoint progress with the completed batch boundary."""
        epoch_loop = getattr(getattr(trainer, "fit_loop", None), "epoch_loop", None)
        batch_progress = getattr(epoch_loop, "batch_progress", None)
        if batch_progress is None:
            return False

        total = getattr(batch_progress, "total", None)
        current = getattr(batch_progress, "current", None)
        if total is None or current is None:
            return False

        should_align = (
            total.processed == total.ready
            and total.completed + 1 == total.processed
            and current.processed == current.ready
            and current.completed + 1 == current.processed
        )
        if not should_align:
            return False

        batch_progress.increment_completed()
        return True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        aligned = self._temporarily_align_completed_for_save(trainer)
        try:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        finally:
            if aligned:
                epoch_loop = getattr(
                    getattr(trainer, "fit_loop", None), "epoch_loop", None
                )
                batch_progress = getattr(epoch_loop, "batch_progress", None)
                if batch_progress is not None:
                    batch_progress.total.completed -= 1
                    batch_progress.current.completed -= 1


class DiskAwareFinalCheckpoint(Callback):
    """Save the final training state while counting it against top-k retention."""

    def __init__(
        self,
        *,
        dirpath,
        model_size,
        context_length,
        token_budget_label="",
        save_top_k,
        monitor="valid_loss",
        mode="min",
        min_free_gb=10,
        cleanup_on_low_space=True,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.model_size = model_size
        self.context_length = context_length
        self.token_budget_label = token_budget_label
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.min_free_gb = min_free_gb
        self.cleanup_on_low_space = cleanup_on_low_space
        self.final_model_path = ""

    def _check_disk_space(self):
        self.dirpath.mkdir(parents=True, exist_ok=True)
        stat = shutil.disk_usage(self.dirpath)
        return stat.free / (1024**3)

    def _monitored_checkpoints(self):
        return [path for path in self.dirpath.glob("s=step=*.ckpt") if path.is_file()]

    def _periodic_checkpoints(self):
        return [path for path in self.dirpath.glob("periodic-*.ckpt") if path.is_file()]

    def _final_checkpoints(self, current_final_path=None):
        return [
            path
            for path in self.dirpath.glob("final-s=step=*.ckpt")
            if path.is_file() and path != current_final_path
        ]

    def _metric_value(self, path):

        matches = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", path.name)
        if not matches:
            return None
        try:
            return float(matches[-1])
        except ValueError:
            return None

    def _sort_monitored_for_retention(self, paths):
        metric_paths = []
        fallback_paths = []
        for path in paths:
            value = self._metric_value(path)
            if value is None:
                fallback_paths.append(path)
            else:
                metric_paths.append((value, path))

        reverse = self.mode == "max"
        metric_paths.sort(key=lambda item: item[0], reverse=reverse)
        fallback_paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return [path for _, path in metric_paths] + fallback_paths

    def _unlink(self, path, reason):
        try:
            path.unlink()
            print(f"[FinalCheckpoint] Removed {reason}: {path.name}")
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[FinalCheckpoint] Failed to remove {path.name}: {exc}")

    def _cleanup_for_retention(self, current_final_path=None):
        if self.save_top_k == -1:
            keep_monitored = None
        else:
            keep_monitored = max(self.save_top_k - 1, 0)

        for path in self._final_checkpoints(current_final_path=current_final_path):
            self._unlink(path, "old final checkpoint")

        if keep_monitored is None:
            return

        monitored = self._sort_monitored_for_retention(self._monitored_checkpoints())
        for path in monitored[keep_monitored:]:
            self._unlink(
                path, f"monitored checkpoint beyond save_top_k_ckpts={self.save_top_k}"
            )

    def _cleanup_for_low_space(self, current_final_path):
        for path in self._periodic_checkpoints():
            self._unlink(path, "periodic checkpoint removed for low disk space")

        self._cleanup_for_retention(current_final_path=current_final_path)

    def _barrier(self, trainer, name):
        strategy = getattr(trainer, "strategy", None)
        if strategy is None or not hasattr(strategy, "barrier"):
            return
        try:
            strategy.barrier(name)
        except TypeError:
            strategy.barrier()

    def on_train_end(self, trainer, pl_module):
        is_global_zero = getattr(trainer, "is_global_zero", True)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        final_step = max(int(getattr(trainer, "global_step", 0)) - 1, 0)
        token_budget_suffix = (
            f"-tokens{self.token_budget_label}" if self.token_budget_label else ""
        )
        final_path = self.dirpath / (
            f"final-s=step={final_step}-{self.model_size}{token_budget_suffix}-ctx{self.context_length}.ckpt"
        )

        free_gb = self._check_disk_space()
        if free_gb < self.min_free_gb:
            message = (
                f"[FinalCheckpoint] Not enough disk space before saving the final checkpoint: "
                f"{free_gb:.1f}GB is below the {self.min_free_gb:.1f}GB threshold at {self.dirpath}."
            )
            if not self.cleanup_on_low_space:
                raise RuntimeError(f"{message} Old checkpoints were not removed.")

            if is_global_zero:
                print(f"{message} Removing periodic checkpoints and checkpoints beyond the retention limit...")
                self._cleanup_for_low_space(final_path)
        self._barrier(trainer, "final_checkpoint_pre_save_cleanup")

        free_gb = self._check_disk_space()
        if free_gb < self.min_free_gb:
            if is_global_zero:
                print(f"[FinalCheckpoint] Free disk space after cleanup: {free_gb:.1f}GB")
            raise RuntimeError(
                "Final checkpoint disk preflight failed: "
                f"free space after cleanup {free_gb:.1f}GB is still below "
                f"--checkpoint_min_free_gb={self.min_free_gb:.1f}GB. "
                f"checkpoint_dir={self.dirpath}"
            )
        if is_global_zero:
            print(f"[FinalCheckpoint] Free disk space before saving final checkpoint: {free_gb:.1f}GB")

        try:
            trainer.save_checkpoint(str(final_path), weights_only=False)
        except TypeError:
            trainer.save_checkpoint(str(final_path))
        self.final_model_path = str(final_path)
        if is_global_zero:
            print(f"[FinalCheckpoint] saved final checkpoint: {final_path}")
            self._cleanup_for_retention(current_final_path=final_path)
        self._barrier(trainer, "final_checkpoint_post_save_cleanup")
