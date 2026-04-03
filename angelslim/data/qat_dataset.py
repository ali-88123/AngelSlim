# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.utils.data import Dataset, IterableDataset

from ..compressor.speculative.train.data.data_utils import (
    paddingtensor2D,
    paddingtensor3D_BHW,
)


class QATDataset(IterableDataset):
    def __init__(self, dataset, tokenizer=None, block_size=2048, is_opensource=False):
        super().__init__()
        if is_opensource:
            self._samples = self._build_from_raw(dataset, tokenizer, block_size)
        else:
            self._samples = self._build_from_internal(dataset)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]

    def __iter__(self):
        return iter(self._samples)

    def _build_from_raw(self, dataset, tokenizer, block_size):
        concatenated = {}
        for sample in dataset:
            tokenized = tokenizer(sample["text"])
            for key in tokenized.keys():
                if key not in concatenated:
                    concatenated[key] = []
                concatenated[key].extend(tokenized[key])

        total = len(concatenated["input_ids"])
        if total >= block_size:
            total = (total // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return [
            {"input_ids": result["input_ids"][i], "labels": result["labels"][i]}
            for i in range(len(result["input_ids"]))
        ]

    def _build_from_internal(self, dataset):
        return [
            {
                "input_ids": dataset[i]["input_ids"].tolist()[0],
                "labels": dataset[i]["labels"].tolist()[0],
            }
            for i in range(len(dataset))
        ]


class BlockTrainDataset(Dataset):
    def __init__(self, size, seqlen, hidden_size, batch_size, dtype):
        self.data = torch.zeros((size // batch_size, batch_size, seqlen, hidden_size), dtype=dtype)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def update_data(self, idx, new_data):
        self.data[idx] = new_data


class VLQATDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        max_length = max(item["input_ids"].shape[1] for item in instances)
        batch_input_ids = torch.cat(
            [paddingtensor2D(item["input_ids"], max_length) for item in instances]
        )
        batch_labels = torch.cat(
            [paddingtensor2D(item["labels"], max_length) for item in instances]
        )
        batch_attention_mask = torch.cat(
            [paddingtensor2D(item["attention_mask"], max_length) for item in instances]
        )
        all_pixel_values, all_image_grid_thw = [], []
        for item in instances:
            all_pixel_values.append(item["pixel_values"])
            all_image_grid_thw.append(item["image_grid_thw"])
        batch_pixel_values = paddingtensor3D_BHW(all_pixel_values)
        batch_image_grid_thw = torch.cat(all_image_grid_thw, dim=0)

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
            "pixel_values": batch_pixel_values,
            "image_grid_thw": batch_image_grid_thw,
        }

        return batch
