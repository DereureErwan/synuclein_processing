# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import math
import os
import random
import string
import time
from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import umap
from lightning.pytorch.callbacks import Callback
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm

import wandb
from solo.utils.misc import gather, omegaconf_select

import numpy as np 

from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, DrawingArea, TextArea, VPacker



class AutoUMAP(Callback):
    def __init__(
        self,
        name: str,
        logdir: Union[str, Path] = Path("auto_umap"),
        frequency: int = 1,
        keep_previous: bool = False,
        color_palette: str = "hls",
    ):
        """UMAP callback that automatically runs UMAP on the validation dataset and uploads the
        figure to wandb.

        Args:
            cfg (DictConfig): DictConfig containing at least an attribute name.
            logdir (Union[str, Path], optional): base directory to store checkpoints.
                Defaults to Path("auto_umap").
            frequency (int, optional): number of epochs between each UMAP. Defaults to 1.
            color_palette (str, optional): color scheme for the classes. Defaults to "hls".
            keep_previous (bool, optional): whether to keep previous plots or not.
                Defaults to False.
        """

        super().__init__()

        self.name = name
        self.logdir = Path(logdir)
        self.frequency = frequency
        self.color_palette = color_palette
        self.keep_previous = keep_previous

    @staticmethod
    def add_and_assert_specific_cfg(cfg: DictConfig) -> DictConfig:
        """Adds specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg.auto_umap = omegaconf_select(cfg, "auto_umap", default={})
        cfg.auto_umap.enabled = omegaconf_select(cfg, "auto_umap.enabled", default=False)
        cfg.auto_umap.dir = omegaconf_select(cfg, "auto_umap.dir", default="auto_umap")
        cfg.auto_umap.frequency = omegaconf_select(cfg, "auto_umap.frequency", default=1)

        return cfg

    @staticmethod
    def random_string(letter_count=4, digit_count=4):
        tmp_random = random.Random(time.time())
        rand_str = "".join(tmp_random.choice(string.ascii_lowercase) for _ in range(letter_count))
        rand_str += "".join(tmp_random.choice(string.digits) for _ in range(digit_count))
        rand_str = list(rand_str)
        tmp_random.shuffle(rand_str)
        return "".join(rand_str)

    def initial_setup(self, trainer: pl.Trainer):
        """Creates the directories and does the initial setup needed.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.logger is None:
            if self.logdir.exists():
                existing_versions = set(os.listdir(self.logdir))
            else:
                existing_versions = []
            version = "offline-" + self.random_string()
            while version in existing_versions:
                version = "offline-" + self.random_string()
        else:
            version = str(trainer.logger.version)
        if version is not None:
            self.path = self.logdir / version
            self.umap_placeholder = f"{self.name}-{version}" + "-ep={}.pdf"
        else:
            self.path = self.logdir
            self.umap_placeholder = f"{self.name}" + "-ep={}.pdf"
        self.last_ckpt: Optional[str] = None

        # create logging dirs
        if trainer.is_global_zero:
            os.makedirs(self.path, exist_ok=True)

    def on_train_start(self, trainer: pl.Trainer, _):
        """Performs initial setup on training start.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        self.initial_setup(trainer)

    def plot(self, trainer: pl.Trainer, module: pl.LightningModule):
        """Produces a UMAP visualization by forwarding all data of the
        first validation dataloader through the module.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
            module (pl.LightningModule): current module object.
        """

        device = module.device
        data = []
        Y = []

        # set module to eval model and collect all feature representations
        module.eval()
        with torch.no_grad():
            val_dataloader = trainer.val_dataloaders
            if isinstance(val_dataloader, list):
                val_dataloader = val_dataloader[0]
            for x, y in val_dataloader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                feats = module(x)["feats"]

                feats = gather(feats)
                y = gather(y)

                data.append(feats.cpu())
                Y.append(y.cpu())
        module.train()

        if trainer.is_global_zero and len(data):
            data = torch.cat(data, dim=0).numpy()
            Y = torch.cat(Y, dim=0)
            num_classes = len(torch.unique(Y))
            Y = Y.numpy()

            data = umap.UMAP(n_components=2).fit_transform(data)

            # passing to dataframe
            df = pd.DataFrame()
            df["feat_1"] = data[:, 0]
            df["feat_2"] = data[:, 1]
            df["Y"] = Y
            plt.figure(figsize=(9, 9))
            ax = sns.scatterplot(
                x="feat_1",
                y="feat_2",
                hue="Y",
                palette=sns.color_palette(self.color_palette, num_classes),
                data=df,
                legend="full",
                alpha=0.3,
            )
            ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
            ax.tick_params(left=False, right=False, bottom=False, top=False)

            # manually improve quality of imagenet umaps
            if num_classes > 100:
                anchor = (0.5, 1.8)
            else:
                anchor = (0.5, 1.35)

            plt.legend(loc="upper center", bbox_to_anchor=anchor, ncol=math.ceil(num_classes / 10))
            plt.tight_layout()

            if isinstance(trainer.logger, pl.loggers.WandbLogger):
                wandb.log(
                    {"validation_umap": wandb.Image(ax)},
                    commit=False,
                )

            # save plot locally as well
            epoch = trainer.current_epoch  # type: ignore
            plt.savefig(self.path / self.umap_placeholder.format(epoch))
            plt.close()

    def on_validation_end(self, trainer: pl.Trainer, module: pl.LightningModule):
        """Tries to generate an up-to-date UMAP visualization of the features
        at the end of each validation epoch.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        epoch = trainer.current_epoch  # type: ignore
        if epoch % self.frequency == 0 and not trainer.sanity_checking:
            self.plot(trainer, module)


class OfflineUMAP:
    def __init__(self, color_palette: str = "hls"):
        """Offline UMAP helper.

        Args:
            color_palette (str, optional): color scheme for the classes. Defaults to "hls".
        """

        self.color_palette = color_palette

    def plot(
        self,
        device: str,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        plot_path: str,
    ):
        """Produces a UMAP visualization by forwarding all data of the
        first validation dataloader through the model.
        **Note: the model should produce features for the forward() function.

        Args:
            device (str): gpu/cpu device.
            model (nn.Module): current model.
            dataloader (torch.utils.data.Dataloader): current dataloader containing data.
            plot_path (str): path to save the figure.
        """

        data = []
        Y = []

        # set module to eval model and collect all feature representations
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Collecting features"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                feats = model(x)
                data.append(feats.cpu())
                Y.append(y.cpu())
        model.train()

        data = torch.cat(data, dim=0).numpy()
        Y = torch.cat(Y, dim=0)
        num_classes = len(torch.unique(Y))
        Y = Y.numpy()

        print("Creating UMAP")
        data = umap.UMAP(n_components=2).fit_transform(data)

        # passing to dataframe
        df = pd.DataFrame()
        df["feat_1"] = data[:, 0]
        df["feat_2"] = data[:, 1]
        df["Y"] = Y
        plt.figure(figsize=(9, 9))
        ax = sns.scatterplot(
            x="feat_1",
            y="feat_2",
            hue="Y",
            palette=sns.color_palette(self.color_palette, num_classes),
            data=df,
            legend=False,   # legend will be handled later
            alpha=0.1,      # faint background
            ax=None
        )

        # Highlight classes 1–4 on top
        highlight_classes = [0, 1, 2, 3, 4, 5]
        markers = ["o", "s", "D", "^", "v", "P"]  # circle, square, diamond, triangle

        for cls, marker in zip(highlight_classes, markers):
            subset = df[df["Y"] == cls]
            sns.scatterplot(
                x="feat_1",
                y="feat_2",
                data=subset,
                color=sns.color_palette(self.color_palette, num_classes)[cls],
                label=f"Class {cls}",
                s=100,           # bigger markers
                alpha=0.9,       # more opaque
                marker=marker,
                ax=ax
            )

        ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
        ax.tick_params(left=False, right=False, bottom=False, top=False)

        # manually improve quality of imagenet umaps
        if num_classes > 100:
            anchor = (0.5, 1.8)
        else:
            anchor = (0.5, 1.35)

        plt.legend(loc="upper center", bbox_to_anchor=anchor, ncol=math.ceil(num_classes / 10))
        plt.tight_layout()

        # save plot locally as well
        plt.savefig(plot_path)
        plt.close()


    def plot_with_thumbnails(
        self,
        device: str,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        plot_path: str,
    ):
        """UMAP with small image thumbnails and label coloring."""

        data = []
        Y = []
        image_paths = list(Path(dataloader.dataset.root).rglob('*.png'))
        # image_paths = [ f for f in os.listdir(dataloader.dataset.root) if os.path.isfile(f) ]
        # print(os.listdir(dataloader.dataset.root)[:10])
        print(image_paths[0])

        model.eval()
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Collecting features"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                feats = model(x)
                data.append(feats.cpu())
                Y.append(y.cpu())

                # Keep original images as PIL for thumbnails

        model.train()

        data = torch.cat(data, dim=0).numpy()
        Y = torch.cat(Y, dim=0).numpy()
        num_classes = len(np.unique(Y))

        # Compute UMAP
        print("Creating UMAP")
        umap_coords = umap.UMAP(n_components=2, random_state=42).fit_transform(data)

        # Normalize to [0, 1] for placing thumbnails
        x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
        y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
        coords_norm = (umap_coords - [x_min, y_min]) / ([x_max - x_min, y_max - y_min])

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_title("UMAP with Image Thumbnails")
        ax.axis("off")

        # Plot background scatter points colored by class
        df = pd.DataFrame()
        df["feat_1"] = coords_norm[:, 0]
        df["feat_2"] = coords_norm[:, 1]
        df["Y"] = Y

        sns.scatterplot(
            x="feat_1",
            y="feat_2",
            hue="Y",
            palette=sns.color_palette(self.color_palette, num_classes),
            data=df,
            alpha=0.3,
            legend=False,
            ax=ax
        )

        max_thumbnails = 2000
        print(data.shape[0])
        if data.shape[0] > max_thumbnails:
            # print(max_thumbnails)
            sampled_indices = np.random.choice(data.shape[0]-1, max_thumbnails, replace=False)
            # print(sampled_indices.shape)
        else:
            sampled_indices = np.arange(data.shape[0])

        print(sampled_indices)

        # Add image thumbnails
        for i in tqdm(sampled_indices, desc="Placing thumbnails"):
            img_path = dataloader.dataset.root / image_paths[i]
            img = Image.open(img_path)
            x, y = coords_norm[i]
            img.thumbnail((16, 16), Image.Resampling.LANCZOS)
            image_box = OffsetImage(img, zoom=0.5)

    # Create a small text label
            # label_box = TextArea(
            #     image_paths[i],  # truncate if needed
            #     textprops=dict(color="black", fontsize=4, ha='center', va='top')
            # )

            # Pack text above image inside a single box
            # box = VPacker(pad=0, sep=1, children=[label_box, image_box], align="center")
            box = VPacker(pad=0, sep=1, children=[image_box], align="center")

            # Add the combined box to the plot
            ab = AnnotationBbox(
                box,
                (coords_norm[i][0], coords_norm[i][1]),
                frameon=False,
                box_alignment=(0.5, 1),  # center align horizontally, top vertically
                zorder=10,
            )
            ax.add_artist(ab)
            # ax.text(x, y - 0.02,  # Slight vertical offset downward
            #     image_paths[i],     # Text to show
            #     fontsize=4,   # Small but visible
            #     ha='center', va='top',
            #     color='black',
            #     zorder=10,
            # )


        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved UMAP with thumbnails to: {plot_path}")