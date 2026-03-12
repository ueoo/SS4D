import os

from abc import abstractmethod
from typing import *

import pandas as pd

from torch.utils.data import Dataset


class StandardVideoDatasetBase(Dataset):
    """
    TRELLIS-style base class for video datasets.

    Args:
        roots (str): comma-separated dataset roots containing metadata.csv
    """

    def __init__(
        self,
        roots: str,
        **kwargs,
    ):
        super().__init__()
        self.roots = roots.split(",")
        self.instances = []
        self.grouped_instances: dict[tuple[str, str], list[str]] = {}
        self.metadata = pd.DataFrame()

        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            metadata = pd.read_csv(os.path.join(root, "metadata.csv"))
            self._stats[key]["Total"] = len(metadata)
            metadata, stats = self.filter_metadata(metadata)
            self._stats[key].update(stats)

            sample_ids, sample_to_frames, group_stats = self.group_instances(metadata)
            self._stats[key].update(group_stats)

            self._stats[key]["Training instances"] = len(sample_ids)
            self.instances.extend([(root, sample_id) for sample_id in sample_ids])
            for sample_id, frames in sample_to_frames.items():
                self.grouped_instances[(root, sample_id)] = frames

            metadata = metadata.copy()
            metadata["sha256"] = metadata["sha256"].astype(str)
            metadata["__root__"] = root
            metadata.set_index(["__root__", "sha256"], inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])

    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass

    def group_instances(self, metadata: pd.DataFrame) -> Tuple[list[str], dict[str, list[str]], dict[str, int]]:
        """
        Group frame-wise rows into sample-wise instances by stripping frame suffix:
          prefix_sampleidx_frameidx -> prefix_sampleidx
        """
        sample_to_frames: dict[str, list[str]] = {}
        for sha256 in metadata["sha256"].astype(str).values:
            sample_id = self.sha_to_sample_id(sha256)
            sample_to_frames.setdefault(sample_id, []).append(sha256)

        for sample_id, frames in sample_to_frames.items():
            sample_to_frames[sample_id] = sorted(frames, key=self.frame_sort_key)

        sample_ids = sorted(sample_to_frames.keys())
        stats = {"Grouped video samples": len(sample_ids)}
        return sample_ids, sample_to_frames, stats

    @abstractmethod
    def get_instance(self, root: str, sample: str) -> Dict[str, Any]:
        pass

    @staticmethod
    def sha_to_sample_id(sha256: str) -> str:
        if "_" not in sha256:
            return sha256
        return sha256.rsplit("_", 1)[0]

    @staticmethod
    def frame_sort_key(sha256: str):
        if "_" not in sha256:
            return (0, sha256)
        prefix, suffix = sha256.rsplit("_", 1)
        try:
            return (int(suffix), prefix)
        except ValueError:
            return (0, sha256)

    def get_sample_frames(self, root: str, sample: str) -> list[str]:
        return self.grouped_instances[(root, sample)]

    def get_metadata_row(self, root: str, sha256: str) -> dict[str, Any]:
        row = self.metadata.loc[(root, sha256)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row.to_dict()

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        root, sample = self.instances[index]
        return self.get_instance(root, sample)

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f"  - Total instances: {len(self)}")
        lines.append(f"  - Sources:")
        for key, stats in self._stats.items():
            lines.append(f"    - {key}:")
            for k, v in stats.items():
                lines.append(f"      - {k}: {v}")
        return "\n".join(lines)
