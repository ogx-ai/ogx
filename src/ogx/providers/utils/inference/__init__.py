# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.models.llama.sku_list import all_registered_models

ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR = {m.huggingface_repo: m.descriptor() for m in all_registered_models()}
