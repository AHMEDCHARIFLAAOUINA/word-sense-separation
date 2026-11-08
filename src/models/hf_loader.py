"""Load a model and its tokenizer from the local model folder."""
# qwen-7b's remote code imports transformers_stream_generator, which breaks under
# transformers 5.x; a stub lets its weights load (geometry). Its forward pass still
# needs qwen7_env (transformers 4.32.0).
import sys as _sys
import types as _types
for _m in ("transformers_stream_generator", "transformers_stream_generator.main"):
    if _m not in _sys.modules:
        _sys.modules[_m] = _types.ModuleType(_m)

import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
)


BASE_MODEL_PATH = os.environ.get("WSS_MODELS", "models")


MODEL_PATHS = {
    # LLaMA models
    "llama-7b": os.path.join(
        BASE_MODEL_PATH,
        "Llama",
        "llama-7b",
    ),
    "llama-2-7b": os.path.join(
        BASE_MODEL_PATH,
        "Llama",
        "Llama-2-7b",
    ),
    "llama-3-8b": os.path.join(
        BASE_MODEL_PATH,
        "Llama",
        "Llama-3-8b",
    ),
    "llama-3.1-8b": os.path.join(
        BASE_MODEL_PATH,
        "Llama",
        "Llama-3.1-8B",
    ),

    # Qwen models
    "qwen-7b": os.path.join(
        BASE_MODEL_PATH,
        "Qwen",
        "qwen-7b",
    ),
    "qwen1.5-7b": os.path.join(
        BASE_MODEL_PATH,
        "Qwen",
        "qwen1.5-7b",
    ),
    "qwen2-7b": os.path.join(
        BASE_MODEL_PATH,
        "Qwen",
        "qwen2-7b",
    ),
    "qwen2.5-7b": os.path.join(
        BASE_MODEL_PATH,
        "Qwen",
        "qwen2.5-7b",
    ),
    "qwen3-8b": os.path.join(
        BASE_MODEL_PATH,
        "Qwen",
        "qwen3-8b",
    ),

    # Encoder models
    "bert-base": os.path.join(
        BASE_MODEL_PATH,
        "Bert",
        "bert-base",
    ),
    "roberta-base": os.path.join(
        BASE_MODEL_PATH,
        "Bert",
        "roberta-base",
    ),

    # Both identifiers point to the same SpanBERT folder.
    "spanbert-base": os.path.join(
        BASE_MODEL_PATH,
        "Bert",
        "spanbert-base-cased",
    ),
    "spanbert-base-cased": os.path.join(
        BASE_MODEL_PATH,
        "Bert",
        "spanbert-base-cased",
    ),

    "xlm-roberta-base": os.path.join(
        BASE_MODEL_PATH,
        "Bert",
        "xlm-roberta-base",
    ),
}


def load_model_and_tokenizer(model_name, device=None):
    """Load a supported model and its tokenizer."""

    if model_name not in MODEL_PATHS:
        available_models = ", ".join(sorted(MODEL_PATHS.keys()))
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {available_models}"
        )

    model_path = MODEL_PATHS[model_name]

    print(f"Loading {model_name}")
    print(f"Path: {model_path}")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model directory does not exist: {model_path}"
        )

    # Original LLaMA-7B
    # Its transferred tokenizer metadata contains empty special-token
    # definitions. The correct standard LLaMA tokens must therefore be
    # supplied explicitly.
    if model_name == "llama-7b":
        tokenizer = LlamaTokenizer.from_pretrained(
            model_path,
            legacy=True,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = LlamaForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto" if device is None else None,
        )

    # LLaMA-2
    # Ships a SentencePiece model, so the LlamaTokenizer class is correct.
    elif model_name == "llama-2-7b":
        tokenizer = LlamaTokenizer.from_pretrained(
            model_path,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = LlamaForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto" if device is None else None,
        )

    # LLaMA-3 and 3.1
    # These ship a byte-level BPE tokenizer in tokenizer.json, not a
    # SentencePiece model. Loading them through LlamaTokenizer discards the
    # space handling: the text does not round-trip and word boundaries fall
    # in the wrong places ("the crane closely" -> th|ec|ran|ec|los|ely).
    # AutoTokenizer selects the correct implementation.
    elif model_name in {"llama-3-8b", "llama-3.1-8b"}:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = LlamaForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto" if device is None else None,
        )

    # Qwen models
    elif model_name.startswith("qwen"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # The old Qwen-7B tokenizer does not define EOS or PAD correctly.
        # Use its existing <|endoftext|> token without adding a new token.
        if model_name == "qwen-7b":
            endoftext_token = "<|endoftext|>"
            endoftext_id = tokenizer.convert_tokens_to_ids(
                endoftext_token
            )

            if endoftext_id is None or endoftext_id < 0:
                raise ValueError(
                    "Qwen-7B tokenizer does not contain "
                    "'<|endoftext|>'."
                )

            if tokenizer.eos_token is None:
                tokenizer.eos_token = endoftext_token

            if tokenizer.pad_token is None:
                tokenizer.pad_token = endoftext_token

        elif tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError(
                    f"{model_name} has neither a pad token "
                    "nor an EOS token."
                )

            tokenizer.pad_token = tokenizer.eos_token

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=torch.float16,
                device_map="auto" if device is None else None,
            )
        except TypeError:
            # Older transformers / legacy remote-code models use torch_dtype
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto" if device is None else None,
            )

    # Encoder-only models
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )

        model = AutoModel.from_pretrained(
            model_path,
        )

    if device is not None:
        model = model.to(device)

    model.eval()

    return tokenizer, model