import glob
import os

import numpy as np
import torch
from cshogi import BLACK, BLACK_WIN, WHITE, WHITE_WIN, Board, HuffmanCodedPosAndEval
from torch.utils.data import DataLoader, Dataset

from gumbel_dlshogi.common import dtypeTrainingData
from gumbel_dlshogi.features import (
    FEATURES1_NUM,
    features2_num,
    make_input_features,
    make_move_label,
    set_nyugyoku_features,
)


def _make_features(board, hcps):
    """Returns the input features x1, x2 and the side to move for the HCP positions.

    The board is reused by overwriting it with each position.
    """
    n = len(hcps)
    # Allocate new arrays each time, since the returned tensors share their memory
    features1 = np.empty((n, FEATURES1_NUM, 9, 9), dtype=np.float32)
    features2 = np.empty((n, features2_num(), 9, 9), dtype=np.float32)
    turns = np.empty(n, dtype=np.int32)
    for i, hcp in enumerate(hcps):
        board.set_hcp(np.asarray(hcp))
        make_input_features(board, features1[i], features2[i])
        turns[i] = board.turn

    return torch.from_numpy(features1), torch.from_numpy(features2), turns


def _make_results(game_results, turns):
    """Converts game results to the values from the side to move."""
    results = np.where(
        game_results == BLACK_WIN,
        turns == BLACK,
        np.where(game_results == WHITE_WIN, turns == WHITE, 0.5),
    )
    return torch.from_numpy(results.astype(np.float32))


def _collate_batch(batch):
    """The datasets build whole batches in __getitems__, so return them as is."""
    return batch


def worker_init_fn(worker_id):
    """Worker initialization function for DataLoader.

    The feature setting is per process, so apply it in each worker.
    """
    dataset = torch.utils.data.get_worker_info().dataset
    set_nyugyoku_features(dataset.use_nyugyoku_features)


class TrainingDataset(Dataset):
    def __init__(self, data_dir, num_files=None, use_nyugyoku_features=False):
        """
        Args:
            data_dir: Directory containing .data files
            num_files: Number of newest files to use (None for all)
            use_nyugyoku_features: Whether to use nyugyoku features
        """
        self.use_nyugyoku_features = use_nyugyoku_features
        # The feature setting is per process, so apply it where the dataset is created.
        set_nyugyoku_features(use_nyugyoku_features)

        # Find and sort data files by modification time (newest first)
        pattern = os.path.join(data_dir, "*.data")
        files = glob.glob(pattern)
        files.sort(key=os.path.getmtime, reverse=True)

        # Select specified number of files
        if num_files is not None:
            files = files[:num_files]

        self.files = files

        # Global index where each file starts (the last element is the total)
        num_samples = [
            os.path.getsize(file) // dtypeTrainingData.itemsize for file in files
        ]
        self.file_starts = np.concatenate([[0], np.cumsum(num_samples, dtype=np.int64)])
        self.total_samples = int(self.file_starts[-1])

        # Memory maps and the board are created lazily in each process
        self._mmaps = {}
        self._board = None

    def __getstate__(self):
        # Do not pickle the memory maps (they would be copied as arrays)
        state = self.__dict__.copy()
        state["_mmaps"] = {}
        state["_board"] = None
        return state

    def _get_mmap(self, file_index):
        """Get or create memory map for a file"""
        if file_index not in self._mmaps:
            self._mmaps[file_index] = np.memmap(
                self.files[file_index], dtype=dtypeTrainingData, mode="r"
            )
        return self._mmaps[file_index]

    def _read_samples(self, indices):
        """Reads the samples at the global indices, keeping their order"""
        if np.any((indices < 0) | (indices >= self.total_samples)):
            raise IndexError(f"Index out of range: {indices}")
        file_indices = np.searchsorted(self.file_starts, indices, side="right") - 1
        if len(indices) > 0 and np.all(file_indices == file_indices[0]):
            # All samples are in one file: advanced indexing already returns a copy
            file_index = file_indices[0]
            return self._get_mmap(file_index)[indices - self.file_starts[file_index]]

        samples = np.empty(len(indices), dtype=dtypeTrainingData)
        for file_index in np.unique(file_indices):
            mask = file_indices == file_index
            local_indices = indices[mask] - self.file_starts[file_index]
            samples[mask] = self._get_mmap(file_index)[local_indices]
        return samples

    def __len__(self):
        return self.total_samples

    def __getitems__(self, indices):
        """Builds a whole batch at once (used by DataLoader)"""
        samples = self._read_samples(np.asarray(indices, dtype=np.int64))

        if self._board is None:
            self._board = Board()
        features1, features2, turns = _make_features(self._board, samples["hcp"])
        policies = torch.from_numpy(np.ascontiguousarray(samples["policy"]))
        results = _make_results(samples["result"], turns)

        return features1, features2, policies, results

    def __getitem__(self, idx):
        return tuple(t[0] for t in self.__getitems__([idx]))

    def __del__(self):
        # Clean up memory maps
        for mmap in self._mmaps.values():
            if hasattr(mmap, "_mmap"):
                mmap._mmap.close()


def create_dataloader(
    data_dir,
    batch_size=32,
    num_files=None,
    num_workers=0,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    use_nyugyoku_features=False,
    **kwargs,
):
    """
    Create a DataLoader for training data

    Args:
        data_dir: Directory containing .data files
        batch_size: Batch size
        num_files: Number of newest files to use (None for all)
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for GPU transfer
        use_nyugyoku_features: Whether to use nyugyoku features
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader instance
    """
    dataset = TrainingDataset(
        data_dir, num_files, use_nyugyoku_features=use_nyugyoku_features
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_batch,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        **kwargs,
    )

    # Log dataset information
    if num_files is not None:
        print(f"Using {len(dataset.files)} files for training:")
        for file in dataset.files:
            print(f" - {file}")

    return dataloader


class TestDataset(Dataset):
    def __init__(self, data_file, use_nyugyoku_features=False):
        """
        Args:
            data_file: Path to binary file containing HuffmanCodedPosAndEval data
            use_nyugyoku_features: Whether to use nyugyoku features
        """
        self.data_file = data_file
        self.use_nyugyoku_features = use_nyugyoku_features
        # The feature setting is per process, so apply it where the dataset is created.
        set_nyugyoku_features(use_nyugyoku_features)

        # Calculate total samples from file size
        file_size = os.path.getsize(data_file)
        self.total_samples = file_size // HuffmanCodedPosAndEval.itemsize

        # The memory map and the board are created lazily in each process
        self._data = None
        self._board = None

    def __getstate__(self):
        # Do not pickle the memory map (it would be copied as an array)
        state = self.__dict__.copy()
        state["_data"] = None
        state["_board"] = None
        return state

    def _get_data(self):
        """Get or create memory map for the data file"""
        if self._data is None:
            self._data = np.memmap(
                self.data_file, dtype=HuffmanCodedPosAndEval, mode="r"
            )
        return self._data

    def __len__(self):
        return self.total_samples

    def __getitems__(self, indices):
        """Builds a whole batch at once (used by DataLoader)"""
        samples = self._get_data()[np.asarray(indices, dtype=np.int64)]

        if self._board is None:
            self._board = Board()
        features1, features2, turns = _make_features(self._board, samples["hcp"])
        labels = torch.tensor(
            [
                make_move_label(int(move16), turn)
                for move16, turn in zip(samples["bestMove16"], turns)
            ],
            dtype=torch.long,
        )
        results = _make_results(samples["gameResult"], turns)

        return features1, features2, labels, results

    def __getitem__(self, idx):
        return tuple(t[0] for t in self.__getitems__([idx]))

    def __del__(self):
        # Clean up memory map
        if (
            hasattr(self, "_data")
            and self._data is not None
            and hasattr(self._data, "_mmap")
        ):
            self._data._mmap.close()


def create_test_dataloader(
    data_file,
    batch_size=32,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
    use_nyugyoku_features=False,
    **kwargs,
):
    """
    Create a DataLoader for test data

    Args:
        data_file: Path to binary file containing HuffmanCodedPosAndEval data
        batch_size: Batch size
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for GPU transfer
        num_workers: Number of worker processes
        use_nyugyoku_features: Whether to use nyugyoku features
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader instance
    """
    dataset = TestDataset(data_file, use_nyugyoku_features=use_nyugyoku_features)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_batch,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        **kwargs,
    )
