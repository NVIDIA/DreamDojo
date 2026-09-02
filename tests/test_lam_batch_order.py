# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for the LAM action batch/time layout.

The full DreamDojo model needs CUDA-only dependencies, so this test keeps the
fixture dependency-free and checks the layout contract directly from the
source.  The unique token fixture makes a batch/time transpose observable even
when tensor shapes remain unchanged.
"""

import ast
from pathlib import Path

import torch


_MODEL_SOURCE = (
    Path(__file__).parents[1]
    / "cosmos_predict2"
    / "_src"
    / "predict2"
    / "models"
    / "text2world_model_rectified_flow.py"
)


def _forward_rearrange_patterns() -> list[str]:
    tree = ast.parse(_MODEL_SOURCE.read_text(encoding="utf-8"))
    forward = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    patterns = []
    for node in ast.walk(forward):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "rearrange" or len(node.args) < 2:
            continue
        pattern = node.args[1]
        if isinstance(pattern, ast.Constant) and isinstance(pattern.value, str):
            patterns.append(pattern.value)
    return patterns


def test_lam_restore_uses_batch_major_inverse():
    """The inverse must match ``(b p)`` flattening used for LAM videos."""

    patterns = _forward_rearrange_patterns()
    assert "b (p t) h w c -> (b p) t h w c" in patterns
    assert "(b t) d -> b t d" in patterns
    assert "(t b) d -> b t d" not in patterns


def test_batch_major_restore_keeps_each_sample_sequence_intact():
    batch_size, action_steps, width = 3, 4, 2
    flattened = torch.arange(batch_size * action_steps * width).reshape(
        batch_size * action_steps, width
    )

    restored = flattened.reshape(batch_size, action_steps, width)
    expected = torch.stack(
        [flattened[s * action_steps : (s + 1) * action_steps] for s in range(batch_size)]
    )

    torch.testing.assert_close(restored, expected)
    # A time-major interpretation has the same shape but mixes samples.
    time_major = flattened.reshape(action_steps, batch_size, width).transpose(0, 1)
    assert not torch.equal(restored, time_major)
