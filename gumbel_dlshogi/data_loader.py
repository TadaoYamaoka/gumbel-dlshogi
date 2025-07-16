import glob
import os

import numpy as np
import torch
from cshogi import BLACK, WHITE, Board, HuffmanCodedPosAndEval
from pydlshogi2.features import FEATURES_NUM, make_input_features, make_move_label
from torch.utils.data import DataLoader, Dataset

from gumbel_dlshogi.common import dtypeTrainingData


class TrainingDataset(Dataset):
    def __init__(self, data_dir, num_files=None, worker_id=None, num_workers=None):
        """
        Args:
            data_dir: Directory containing .data files
            num_files: Number of newest files to use (None for all)
            worker_id: Worker ID for multi-worker setup
            num_workers: Total number of workers
        """
        self.data_dir = data_dir
        self.num_files = num_files
        self.worker_id = worker_id or 0
        self.num_workers = num_workers or 1

        # Find and sort data files by modification time (newest first)
        pattern = os.path.join(data_dir, "*.data")
        files = glob.glob(pattern)
        files.sort(key=os.path.getmtime, reverse=True)

        # Select specified number of files
        if num_files is not None:
            files = files[:num_files]

        self.files = files
        self.file_offsets = []
        self.total_samples = 0

        # Calculate file offsets and total samples
        for file_path in self.files:
            file_size = os.path.getsize(file_path)
            num_samples = file_size // dtypeTrainingData.itemsize
            self.file_offsets.append((self.total_samples, num_samples, file_path))
            self.total_samples += num_samples

        # For multi-worker setup, divide data among workers
        if self.num_workers > 1:
            samples_per_worker = self.total_samples // self.num_workers
            self.start_idx = self.worker_id * samples_per_worker
            if self.worker_id == self.num_workers - 1:
                # Last worker takes remaining samples
                self.end_idx = self.total_samples
            else:
                self.end_idx = (self.worker_id + 1) * samples_per_worker
            self.worker_total_samples = self.end_idx - self.start_idx
        else:
            self.start_idx = 0
            self.end_idx = self.total_samples
            self.worker_total_samples = self.total_samples

        # Create memory maps for files
        self._mmaps = {}

        self._board = None

    def _get_mmap(self, file_path):
        """Get or create memory map for a file"""
        if file_path not in self._mmaps:
            self._mmaps[file_path] = np.memmap(
                file_path, dtype=dtypeTrainingData, mode="r"
            )
        return self._mmaps[file_path]

    def _find_file_and_offset(self, global_idx):
        """Find which file contains the sample at global_idx"""
        for file_start, file_samples, file_path in self.file_offsets:
            if global_idx < file_start + file_samples:
                return file_path, global_idx - file_start
        raise IndexError(f"Index {global_idx} out of range")

    def _get_board(self):
        """Get or create a Board instance for HCP conversion"""
        if self._board is None:
            self._board = Board()
        return self._board

    def __len__(self):
        return self.worker_total_samples

    def __getitem__(self, idx):
        if idx >= self.worker_total_samples:
            raise IndexError(f"Index {idx} out of range for worker {self.worker_id}")

        # Convert worker-local index to global index
        global_idx = self.start_idx + idx

        # Find file and local offset
        file_path, local_offset = self._find_file_and_offset(global_idx)

        # Get memory map and read sample
        mmap = self._get_mmap(file_path)
        sample = mmap[local_offset]

        # Convert HCP to input features
        board = self._get_board()
        board.set_hcp(np.asarray(sample["hcp"]))

        features = np.empty((FEATURES_NUM, 9, 9), dtype=np.float32)
        make_input_features(board, features)

        match sample["result"]:
            case 1:  # BLACK_WIN
                result = 1 if board.turn == BLACK else 0
            case 2:  # WHITE_WIN
                result = 1 if board.turn == WHITE else 0
            case _:  # DRAW
                result = 0.5

        # Convert to tensors
        features_tensor = torch.from_numpy(features)
        policy_tensor = torch.from_numpy(sample["policy"].copy())
        result_tensor = torch.tensor(result, dtype=torch.float32)

        return features_tensor, policy_tensor, result_tensor

    def __del__(self):
        # Clean up memory maps
        for mmap in self._mmaps.values():
            if hasattr(mmap, "_mmap"):
                mmap._mmap.close()


def worker_init_fn(worker_id):
    """
    Worker initialization function for DataLoader.
    Re-initializes the dataset for each worker process.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # The dataset instance in this worker.

    worker_dataset = TrainingDataset(
        data_dir=dataset.data_dir,
        num_files=dataset.num_files,
        worker_id=worker_id,
        num_workers=dataset.num_workers,
    )

    # Copy the attributes from the new dataset to the one in the worker
    dataset.files = worker_dataset.files
    dataset.file_offsets = worker_dataset.file_offsets
    dataset.total_samples = worker_dataset.total_samples
    dataset.worker_id = worker_dataset.worker_id
    dataset.start_idx = worker_dataset.start_idx
    dataset.end_idx = worker_dataset.end_idx
    dataset.worker_total_samples = worker_dataset.worker_total_samples
    dataset._mmaps = {}  # Each worker needs its own memory maps
    dataset._board = None  # Reset the board for each worker


def create_dataloader(
    data_dir,
    batch_size=32,
    num_files=None,
    num_workers=0,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
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
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader instance
    """

    if num_workers > 0:
        # Create a dummy dataset for the main process
        dataset = TrainingDataset(data_dir, num_files)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
            **kwargs,
        )
    else:
        dataset = TrainingDataset(data_dir, num_files)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs,
        )

    return dataloader


class TestDataset(Dataset):
    def __init__(self, data_file, worker_id=None, num_workers=None):
        """
        Args:
            data_file: Path to binary file containing HuffmanCodedPosAndEval data
            worker_id: Worker ID for multi-worker setup
            num_workers: Total number of workers
        """
        self.data_file = data_file
        self.worker_id = worker_id or 0
        self.num_workers = num_workers or 1

        # Calculate total samples from file size
        file_size = os.path.getsize(data_file)
        self.total_samples = file_size // HuffmanCodedPosAndEval.itemsize

        # For multi-worker setup, divide data among workers
        if self.num_workers > 1:
            samples_per_worker = self.total_samples // self.num_workers
            self.start_idx = self.worker_id * samples_per_worker
            if self.worker_id == self.num_workers - 1:
                # Last worker takes remaining samples
                self.end_idx = self.total_samples
            else:
                self.end_idx = (self.worker_id + 1) * samples_per_worker
            self.worker_total_samples = self.end_idx - self.start_idx
        else:
            self.start_idx = 0
            self.end_idx = self.total_samples
            self.worker_total_samples = self.total_samples

        self._data = None  # Lazy initialization of memory map
        self._board = None

    def _get_data(self):
        """Get or create memory map for the data file"""
        if self._data is None:
            self._data = np.memmap(
                self.data_file, dtype=HuffmanCodedPosAndEval, mode="r"
            )
        return self._data

    def _get_board(self):
        """Get or create a Board instance for HCP conversion"""
        if self._board is None:
            self._board = Board()
        return self._board

    def __len__(self):
        return self.worker_total_samples

    def __getitem__(self, idx):
        if idx >= self.worker_total_samples:
            raise IndexError(f"Index {idx} out of range for worker {self.worker_id}")

        # Convert worker-local index to global index
        global_idx = self.start_idx + idx

        # Get memory map and read sample
        data = self._get_data()
        sample = data[global_idx]

        # Convert HCP to input features
        board = self._get_board()
        board.set_hcp(np.asarray(sample["hcp"]))

        features = np.empty((FEATURES_NUM, 9, 9), dtype=np.float32)
        make_input_features(board, features)

        # Convert best move to policy label
        policy_label = make_move_label(sample["bestMove16"], board.turn)

        # Convert game result to value
        match sample["gameResult"]:
            case 1:  # BLACK_WIN
                result = 1 if board.turn == BLACK else 0
            case 2:  # WHITE_WIN
                result = 1 if board.turn == WHITE else 0
            case _:  # DRAW
                result = 0.5

        # Convert to tensors
        features_tensor = torch.from_numpy(features)
        policy_tensor = torch.tensor(policy_label, dtype=torch.long)
        result_tensor = torch.tensor(result, dtype=torch.float32)

        return features_tensor, policy_tensor, result_tensor

    def __del__(self):
        # Clean up memory map
        if (
            hasattr(self, "_data")
            and self._data is not None
            and hasattr(self._data, "_mmap")
        ):
            self._data._mmap.close()


def test_worker_init_fn(worker_id):
    """
    Worker initialization function for test DataLoader.
    Re-initializes the dataset for each worker process.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # The dataset instance in this worker.

    worker_dataset = TestDataset(
        data_file=dataset.data_file,
        worker_id=worker_id,
        num_workers=dataset.num_workers,
    )

    # Copy the attributes from the new dataset to the one in the worker
    dataset.worker_id = worker_dataset.worker_id
    dataset.start_idx = worker_dataset.start_idx
    dataset.end_idx = worker_dataset.end_idx
    dataset.worker_total_samples = worker_dataset.worker_total_samples
    dataset.total_samples = worker_dataset.total_samples
    dataset._data = None  # Each worker needs its own memory map
    dataset._board = None  # Reset the board for each worker


def create_test_dataloader(
    data_file,
    batch_size=32,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
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
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader instance
    """
    if num_workers > 0:
        # Create a dummy dataset for the main process
        dataset = TestDataset(data_file)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=test_worker_init_fn,
            **kwargs,
        )
    else:
        dataset = TestDataset(data_file)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            **kwargs,
        )

    return dataloader
