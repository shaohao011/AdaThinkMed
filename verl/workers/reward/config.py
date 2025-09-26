# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Reward config
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from ..actor.config import FSDPConfig, ModelConfig, OffloadConfig, OptimConfig


@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    # override_config: Dict[str, Any] = field(default_factory=dict)
    # enable_gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    # freeze_vision_tower: bool = False

    def post_init(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        if self.model_path is not None and os.path.exists(self.model_path):  # ray job uses absolute path
            self.model_path = os.path.abspath(self.model_path)

        if self.tokenizer_path is not None and os.path.exists(self.tokenizer_path):
            self.tokenizer_path = os.path.abspath(self.tokenizer_path)


@dataclass
class RewardConfig:
    reward_type: str = "batch"
    model: ModelConfig = field(default_factory=ModelConfig)
    reward_function: Optional[str] = None
    reward_function_kwargs: dict = field(default_factory=dict)
    skip_special_tokens: bool = True
    num_cpus: int = 1
    length_reward_method: str = "FAST"
    # below are auto keys
    reward_function_name: Optional[str] = field(default=None, init=False)
    ## length reward related:
    difficulty_threshold: float = 0.8
    entropy_topk: float = 0.2
    L_avg_moment: float = 0.9
    alpha: float = 1.0
    beta: float = 1.0
    ulysses_size: int = 1
    global_batch_size: int = 128
    
    norm_method: str = "log_minmax"
    method_version: str = "v6"
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    

    def post_init(self):
        if self.reward_function is not None:  # support custom reward function, e.g., ./math.py:main
            if ":" not in self.reward_function:
                self.reward_function_name = "main"
            else:
                self.reward_function, self.reward_function_name = self.reward_function.rsplit(":", maxsplit=1)

            if os.path.exists(self.reward_function):  # ray job uses absolute path
                self.reward_function = os.path.abspath(self.reward_function)
            else:
                print(f"Reward function {self.reward_function} not found.")
                self.reward_function = None
