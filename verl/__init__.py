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

import importlib
import logging
import os

from packaging.version import parse as parse_version

from .protocol import DataProto
from .utils.device import is_npu_available
from .utils.import_utils import import_external_libs
from .utils.logging_utils import set_basic_config

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()


set_basic_config(level=logging.WARNING)


__all__ = ["DataProto", "__version__"]


modules = os.getenv("VERL_USE_EXTERNAL_MODULES", "")
if modules:
    modules = modules.split(",")
    import_external_libs(modules)


if os.getenv("VERL_USE_MODELSCOPE", "False").lower() == "true":
    if importlib.util.find_spec("modelscope") is None:
        raise ImportError("You are using the modelscope hub, please install modelscope by `pip install modelscope -U`")

    from modelscope.utils.hf_util import patch_hub

    patch_hub()


if is_npu_available:










    import torch

    try:
        if hasattr(torch.nested.nested_tensor, "__wrapped__"):
            torch.nested.nested_tensor = torch.nested.nested_tensor.__wrapped__
        if hasattr(torch.nested.as_nested_tensor, "__wrapped__"):
            torch.nested.as_nested_tensor = torch.nested.as_nested_tensor.__wrapped__
    except AttributeError:
        pass


    from .models.transformers import npu_patch as npu_patch  # noqa












    import tensordict

    if parse_version(tensordict.__version__) < parse_version("0.10.0"):
        from tensordict.base import TensorDictBase

        def _sync_all_patch(self):
            from torch._utils import _get_available_device_type, _get_device_module

            device_type = _get_available_device_type()
            if device_type is None:
                return

            device_module = _get_device_module(device_type)
            device_module.synchronize()

        TensorDictBase._sync_all = _sync_all_patch
