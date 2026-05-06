# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# (Required) NeMo Microservices URLs
NDS_URL = "http://data-store.test:3000" # Data Store
NEMO_URL = "http://nemo.test:3000" # Customizer, Evaluator, Guardrails
NIM_URL = "http://nim.test:3000" # NIM

# (Required) Configure the base model. Must be one supported by the NeMo Customizer deployment!
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# (Required) Hugging Face Token
HF_TOKEN = ""

# (Optional) Namespace to associate with Datasets and Customization jobs
NAMESPACE = "nvidia-e2e-tutorial"

# (Optional) Entity Store Project ID. Modify if you've created a project in Entity Store that you'd
# like to associate with your Customized models.
PROJECT_ID = ""

# (Optional) Directory to save the Customized model
CUSTOMIZED_MODEL_DIR = "nvidia-e2e-tutorial/test-ogx@v1"
