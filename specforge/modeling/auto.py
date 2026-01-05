import json
import os
from typing import Optional, Union

import torch
from SpecForge.specforge.utils import print_on_rank0
from transformers import AutoConfig
from transformers import AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import (
    GptOssConfig,
    Llama4Config,
    Llama4TextConfig,
    LlamaConfig,
    Phi3Config,
    PretrainedConfig,
    Qwen2Config,
    Qwen3Config,
    Qwen3MoeConfig,
    modeling_utils,
)

from .draft.llama3_eagle import LlamaForCausalLMEagle3
from .draft.eagle3 import EagleForCausalLM
from .target.custom_backend import (
    GptOssForCausalLM,
    Llama4ForCausalLM,
    LlamaForCausalLM,
    Phi3ForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)


class AutoEagle3DraftModel(AutoModelForCausalLMBase):
    # the model mapping is currently hardcoded, we should support lazy model mapping via registry
    _model_mapping = {
        LlamaConfig: LlamaForCausalLMEagle3,
        # LlamaConfig: EagleForCausalLM
    }

    @classmethod
    def from_config(cls, config: PretrainedConfig, torch_dtype=None, **config_kwargs):
        """
        This class method takes a configuration object and create its model based on the
        _model_mapping class variable.

        Args:
            config (PretrainedConfig): A configuration object.

        Returns:
            A model instance.
        """
        print_on_rank0(f"AutoEagle3DraftModel.from_config: config type={type(config).__name__}")
        print_on_rank0(f"Model class: {cls._model_mapping[type(config)]}")
        # get the model class from the
        _model_cls = cls._model_mapping[type(config)]
        print_on_rank0(f"Creating model instance with config_kwargs: {list(config_kwargs.keys())}")
        model = _model_cls(config, **config_kwargs)
        print_on_rank0("Model instance created successfully")

        # Convert model to specified dtype if provided
        if torch_dtype is not None:
            print_on_rank0(f"Converting model to dtype: {torch_dtype}")
            model = model.to(dtype=torch_dtype)
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        *model_args,
        **kwargs,
    ):
        print_on_rank0(f"AutoEagle3DraftModel.from_pretrained: loading from {pretrained_model_name_or_path}")
        print_on_rank0(f"Additional kwargs: {list(kwargs.keys())}")
        
        original_warn = modeling_utils.logger.warning

        def filtered_warning(msg):
            if "embed_tokens.weight" in str(msg) and "initialized" in str(msg):
                return
            original_warn(msg)

        modeling_utils.logger.warning = filtered_warning

        try:
            print_on_rank0("Loading model from pretrained...")
            model = super().from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
            print_on_rank0("Model loaded from pretrained successfully")
        finally:
            modeling_utils.logger.warning = original_warn

        return model


class AutoDistributedTargetModel(AutoModelForCausalLMBase):
    # the model mapping is currently hardcoded, we should support lazy model mapping via registry
    _model_mapping = {
        Llama4TextConfig: [Llama4ForCausalLM],
        Qwen3MoeConfig: [Qwen3MoeForCausalLM],
        Qwen2Config: [Qwen2ForCausalLM],
        LlamaConfig: [LlamaForCausalLM],
        Qwen3Config: [Qwen3ForCausalLM],
        Phi3Config: [Phi3ForCausalLM],
        GptOssConfig: [GptOssForCausalLM],
    }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **config_kwargs,
    ):
        print_on_rank0(f"AutoDistributedTargetModel.from_pretrained: loading from {pretrained_model_name_or_path}")
        print_on_rank0(f"Device: {device}, torch_dtype: {torch_dtype}, cache_dir: {cache_dir}")
        
        print_on_rank0("Loading config...")
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
        )
        print_on_rank0(f"Config loaded: type={type(config).__name__}")

        if isinstance(config, Llama4Config):
            print_on_rank0("Llama4Config detected, extracting text_config")
            config = config.text_config

        assert (
            type(config) in cls._model_mapping
        ), f"Unsupported config type: {type(config)}"
        model_cls = cls._model_mapping[type(config)][0]
        print_on_rank0(f"Using model class: {model_cls.__name__}")
        print_on_rank0("Loading model from pretrained...")
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            **config_kwargs,
        )
        print_on_rank0("Model loaded successfully")

        if device is not None:
            print_on_rank0(f"Moving model to device: {device}")
            model = model.to(device)
        else:
            print_on_rank0("Moving model to CUDA")
            model = model.cuda()
        print_on_rank0("Model moved to target device")
        return model


class AutoDraftModelConfig:

    _config_mapping = {
        "LlamaForCausalLMEagle3": LlamaConfig,
        # "EagleForCausalLM": LlamaConfig
    }

    @classmethod
    def from_file(cls, config_path: str):
        """
        This class method takes a configuration file path and create its configuration object based on the
        _config_mapping class variable.

        Args:
            config_path (str): A path to a configuration file.

        Returns:
            A configuration object.
        """
        print_on_rank0(f"AutoDraftModelConfig.from_file: loading config from {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
        print_on_rank0("Config file loaded successfully")

        if "tie_word_embeddings" in config:
            print_on_rank0("Set draft model tie_word_embeddings to False")
            config["tie_word_embeddings"] = False

        # check for architectures
        architectures = config.get("architectures", None)
        print_on_rank0(f"Found architectures: {architectures}")

        if architectures is None:
            raise ValueError("No architectures found in the config file")

        if len(architectures) != 1:
            raise ValueError("Only one architecture is supported")

        architecture = architectures[0]
        print_on_rank0(f"Using architecture: {architecture}")

        if architecture not in cls._config_mapping:
            raise ValueError(f"Architecture {architecture} not supported")

        print_on_rank0(f"Mapping architecture to config class: {cls._config_mapping[architecture]}")

        # If draft_vocab_size is not in config or is None, set draft_vocab_size to vocab_size
        if "draft_vocab_size" not in config or config["draft_vocab_size"] is None:
            vocab_size = config.get("vocab_size", None)
            config["draft_vocab_size"] = vocab_size
            print_on_rank0(f"Set draft_vocab_size to vocab_size: {vocab_size}")

        print_on_rank0("Creating config object from dict...")
        config_obj = cls._config_mapping[architecture].from_dict(config)
        print_on_rank0("Config object created successfully")
        return config_obj