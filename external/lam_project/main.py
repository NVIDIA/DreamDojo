from lightning.pytorch.cli import LightningCLI

from lam.dataset import LightningVideoDataset
from lam.model import LAM


def main() -> None:
    LightningCLI(
        LAM,
        LightningVideoDataset,
        seed_everything_default=32,
    )


if __name__ == "__main__":
    main()
