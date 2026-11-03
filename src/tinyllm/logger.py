"""Experiment logging behind one small interface (D-013).

The training loop only calls log / log_sample / finish. Switching backend (W&B now,
TensorBoard as the fallback) means adding one class here. Only scalars and text
samples are logged, never checkpoints (W&B Free plan: 5 GB/month).

W&B login happens once, inside WSL: `uv run wandb login`. The key goes to ~/.netrc,
never into the repo.
"""


class Logger:
    """No-op logger (logger="none"): used by the tests and for quick local runs."""

    run_id: str | None = None

    def log(self, metrics: dict[str, float], step: int) -> None:
        pass

    def log_sample(self, text: str, step: int) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbLogger(Logger):
    def __init__(self, project: str, name: str, config: dict, mode: str = "online",
                 run_id: str | None = None, dir: str | None = None):
        import wandb  # imported lazily: the "none" logger never needs it

        self._wandb = wandb
        # Passing the stored run id on resume continues the same W&B run (same charts).
        self.run = wandb.init(project=project, name=name, config=config, mode=mode,
                              id=run_id, resume="allow" if run_id else None, dir=dir)
        self.run_id = self.run.id
        self._samples: list[list] = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.run.log(metrics, step=step)

    def log_sample(self, text: str, step: int) -> None:
        # W&B tables are immutable, so re-log the growing table: one row per eval.
        self._samples.append([step, text])
        self.run.log({"samples": self._wandb.Table(columns=["step", "text"], data=self._samples)},
                     step=step)

    def finish(self) -> None:
        self.run.finish()


def make_logger(backend: str, **kwargs) -> Logger:
    """kwargs are WandbLogger arguments; the no-op logger ignores them."""
    if backend == "none":
        return Logger()
    if backend == "wandb":
        return WandbLogger(**kwargs)
    raise ValueError(f"unknown logger {backend!r}; use 'wandb' or 'none'")
