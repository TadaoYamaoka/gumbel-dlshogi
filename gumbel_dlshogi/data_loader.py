import glob
import os

import numpy as np
import torch
from cshogi import BLACK, BLACK_WIN, WHITE, WHITE_WIN, Board, HuffmanCodedPosAndEval
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from gumbel_dlshogi.common import (
    POLICY_FILE_EXT,
    POSITION_FILE_EXT,
    dtypeMovePolicy,
    dtypePosition,
)
from gumbel_dlshogi.features import (
    FEATURES1_NUM,
    MOVE_LABELS_NUM,
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
            data_dir: Directory containing the position (.pos) and policy (.policy)
                files written by self-play
            num_files: Number of newest files to use (None for all)
            use_nyugyoku_features: Whether to use nyugyoku features
        """
        self.use_nyugyoku_features = use_nyugyoku_features
        # The feature setting is per process, so apply it where the dataset is created.
        set_nyugyoku_features(use_nyugyoku_features)

        # Find and sort position files by modification time (newest first)
        pattern = os.path.join(data_dir, "*" + POSITION_FILE_EXT)
        files = [file for file in glob.glob(pattern) if os.path.getsize(file) > 0]
        files.sort(key=os.path.getmtime, reverse=True)

        # Select specified number of files
        if num_files is not None:
            files = files[:num_files]

        self.files = files

        # Offset of each position's moves in the policy file (the last element is
        # the total), computed from the numbers of moves
        self.move_offsets = []
        for file in files:
            # np.fromfile silently drops a truncated last record, so check the size
            if os.path.getsize(file) % dtypePosition.itemsize:
                raise ValueError(f"Invalid position file size: {file}")
            num_moves = np.fromfile(file, dtype=dtypePosition)["num_moves"]
            move_offsets = np.concatenate([[0], np.cumsum(num_moves, dtype=np.int64)])
            policy_size = os.path.getsize(self._policy_file(file))
            if move_offsets[-1] * dtypeMovePolicy.itemsize != policy_size:
                raise ValueError(f"The policy file does not match {file}")
            self.move_offsets.append(move_offsets)

        # Global index where each file starts (the last element is the total)
        num_samples = [len(move_offsets) - 1 for move_offsets in self.move_offsets]
        self.file_starts = np.concatenate([[0], np.cumsum(num_samples, dtype=np.int64)])
        self.total_samples = int(self.file_starts[-1])

        # Memory maps and the board are created lazily in each process
        self._mmaps = {}
        self._board = None

    @staticmethod
    def _policy_file(position_file):
        return os.path.splitext(position_file)[0] + POLICY_FILE_EXT

    def __getstate__(self):
        # Do not pickle the memory maps (they would be copied as arrays)
        state = self.__dict__.copy()
        state["_mmaps"] = {}
        state["_board"] = None
        return state

    def _get_mmaps(self, file_index):
        """Get or create memory maps for the position and policy files"""
        if file_index not in self._mmaps:
            file = self.files[file_index]
            self._mmaps[file_index] = (
                np.memmap(file, dtype=dtypePosition, mode="r"),
                np.memmap(self._policy_file(file), dtype=dtypeMovePolicy, mode="r"),
            )
        return self._mmaps[file_index]

    def _read_samples(self, indices):
        """Reads the positions and policies at the indices, keeping their order"""
        if np.any((indices < 0) | (indices >= self.total_samples)):
            raise IndexError(f"Index out of range: {indices}")
        file_indices = np.searchsorted(self.file_starts, indices, side="right") - 1

        positions = np.empty(len(indices), dtype=dtypePosition)
        # Only moves with non-zero probability are stored; the others stay zero
        policies = np.zeros((len(indices), MOVE_LABELS_NUM), dtype=np.float32)
        for file_index in np.unique(file_indices):
            rows = np.flatnonzero(file_indices == file_index)
            local_indices = indices[rows] - self.file_starts[file_index]
            position_mmap, policy_mmap = self._get_mmaps(file_index)
            positions[rows] = position_mmap[local_indices]

            # Gather the stored moves of each position and scatter them to its row
            move_offsets = self.move_offsets[file_index]
            # starts: positions in the policy file, batch_starts: positions in moves
            starts = move_offsets[local_indices]
            counts = move_offsets[local_indices + 1] - starts
            batch_starts = np.cumsum(counts) - counts
            move_indices = np.arange(counts.sum()) + np.repeat(
                starts - batch_starts, counts
            )
            moves = policy_mmap[move_indices]
            policies[np.repeat(rows, counts), moves["label"]] = moves["prob"]
        return positions, policies

    def __len__(self):
        return self.total_samples

    def __getitems__(self, indices):
        """Builds a whole batch at once (used by DataLoader)"""
        positions, policies = self._read_samples(np.asarray(indices, dtype=np.int64))

        if self._board is None:
            self._board = Board()
        features1, features2, turns = _make_features(self._board, positions["hcp"])
        results = _make_results(positions["result"], turns)

        return features1, features2, torch.from_numpy(policies), results

    def __getitem__(self, idx):
        return tuple(t[0] for t in self.__getitems__([idx]))

    def __del__(self):
        # Clean up memory maps (they do not exist if __init__ raised)
        for mmaps in getattr(self, "_mmaps", {}).values():
            for mmap in mmaps:
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
    num_replicas=1,
    rank=0,
    seed=0,
    **kwargs,
):
    """
    Create a DataLoader for training data

    Args:
        data_dir: Directory containing the self-play position and policy files
        batch_size: Batch size
        num_files: Number of newest files to use (None for all)
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for GPU transfer
        use_nyugyoku_features: Whether to use nyugyoku features
        num_replicas: Number of processes of data parallel training. If more than 1,
            each process loads its own part of the data by DistributedSampler, whose
            set_epoch() must be called at each epoch to shuffle differently.
        rank: Rank of the process in data parallel training
        seed: Seed of the shuffle in data parallel training, the same in all processes
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader instance
    """
    dataset = TrainingDataset(
        data_dir, num_files, use_nyugyoku_features=use_nyugyoku_features
    )
    sampler = None
    if num_replicas > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        shuffle = False
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
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
