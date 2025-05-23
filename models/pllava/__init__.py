# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# from .modeling_pllava_flow import PllavaFlowForConditionalGeneration
from .modeling_pllava import PllavaForConditionalGeneration
from .modeling_pllava_SF import PllavaSFForConditionalGeneration
from .processing_pllava import PllavaProcessor
from .configuration_pllava import PllavaConfig

# _import_structure = {"configuration_pllava": ["PLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP", "PllavaConfig"]}

# try:
#     if not is_torch_available():
#         raise OptionalDependencyNotAvailable()
# except OptionalDependencyNotAvailable:
#     pass
# else:
#     _import_structure["modeling_pllava"] = [
#         "PLLAVA_PRETRAINED_MODEL_ARCHIVE_LIST",
#         "PllavaForConditionalGeneration",
#         "PllavaPreTrainedModel",
#     ]
#     _import_structure["processing_pllava"] = ["PllavaProcessor"]


# if TYPE_CHECKING:
#     from .configuration_pllava import PLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP, PllavaConfig

#     try:
#         if not is_torch_available():
#             raise OptionalDependencyNotAvailable()
#     except OptionalDependencyNotAvailable:
#         pass
#     else:
#         from .modeling_pllava import (
#             PLLAVA_PRETRAINED_MODEL_ARCHIVE_LIST,
#             PllavaForConditionalGeneration,
#             PllavaPreTrainedModel,
#         )
#         from .processing_pllava import PllavaProcessor


# else:
#     import sys

#     sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
