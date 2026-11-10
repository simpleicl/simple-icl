from typing import Iterable, Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        bucket_ids: Iterable[int] | None = None,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        if self.batch_size < 1:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        if self.num_processes < 1:
            raise ValueError(f"`num_processes` must be positive, got {num_processes}")
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        self._bucket_groups = None
        if bucket_ids is not None:
            if torch.is_tensor(bucket_ids):
                bucket_ids = bucket_ids.to(dtype=torch.int64, device="cpu").flatten()
            else:
                bucket_ids = torch.as_tensor(list(bucket_ids), dtype=torch.int64)
            if bucket_ids.numel() != len(dataset):
                raise ValueError(
                    "`bucket_ids` must contain one value per dataset sample, "
                    f"got {bucket_ids.numel()} for dataset length {len(dataset)}."
                )
            # Keep one sorted index tensor rather than a Python list of every
            # sample. This is substantially smaller for large transition sets.
            sorted_bucket_ids, sorted_indices = torch.sort(bucket_ids)
            _, counts = torch.unique_consecutive(sorted_bucket_ids, return_counts=True)
            groups = []
            start = 0
            for count in counts.tolist():
                end = start + int(count)
                groups.append(sorted_indices[start:end])
                start = end
            self._bucket_groups = groups

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        if self._bucket_groups is None:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            # Complete batches stay inside one length bucket. Shuffle those
            # batches globally so bucket lengths are interleaved, then defer
            # remainders to the end so only a few tail batches mix lengths.
            bucket_order = torch.randperm(len(self._bucket_groups), generator=g).tolist()
            complete_batches = []
            remainder = []
            for bucket_index in bucket_order:
                bucket = self._bucket_groups[bucket_index]
                shuffled = bucket[torch.randperm(len(bucket), generator=g)].tolist()
                full_size = (len(shuffled) // self.batch_size) * self.batch_size
                complete_batches.extend(
                    shuffled[start : start + self.batch_size]
                    for start in range(0, full_size, self.batch_size)
                )
                remainder.extend(shuffled[full_size:])
            batch_order = torch.randperm(len(complete_batches), generator=g).tolist()
            indices = [
                index
                for batch_index in batch_order
                for index in complete_batches[batch_index]
            ]
            if remainder:
                remainder = [
                    remainder[index]
                    for index in torch.randperm(len(remainder), generator=g).tolist()
                ]
                indices.extend(remainder)
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)
