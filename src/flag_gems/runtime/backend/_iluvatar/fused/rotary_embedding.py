# Copyright 2026 FlagOS Contributors
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

"""Iluvatar specialization; unsupported inputs keep the generic implementation."""
import torch

from .rotary_embedding_parallel import rope_parallel


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None,
                         rotary_interleaved=False, inplace=False):
    supported = (
        q.ndim == k.ndim == 3
        and q.shape[0] == k.shape[0]
        and q.shape[-1] == k.shape[-1] == 128
        and q.dtype == k.dtype and q.dtype in (torch.float16, torch.bfloat16)
        and q.device.type == "cuda"
        and q.device == k.device == cos.device == sin.device
        and cos.ndim == sin.ndim == 2
        and cos.shape == sin.shape and cos.shape[-1] == 64
        and cos.stride(-1) == sin.stride(-1) == 1
        and position_ids is not None
        and position_ids.ndim == 1 and position_ids.shape[0] == q.shape[0]
        and position_ids.device == q.device
        and position_ids.dtype in (torch.int32, torch.int64)
        and q.stride(-1) == k.stride(-1) == 1
        and q.stride(-2) >= 128 and k.stride(-2) >= 128
        and q.stride(0) >= q.shape[1] * q.stride(1)
        and k.stride(0) >= k.shape[1] * k.stride(1)
    )
    if supported:
        return rope_parallel(q, k, cos, sin, position_ids,
                             interleaved=rotary_interleaved, inplace=inplace)
    from flag_gems.fused.rotary_embedding import apply_rotary_pos_emb as generic
    return generic(q, k, cos, sin, position_ids, rotary_interleaved, inplace)
