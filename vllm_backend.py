"""Independent candidate readout using vLLM's generic raw embedding runtime."""
import importlib.metadata
import json
import os

from gemma_backend import GemmaDecisions, selected_logits


def checkpoint_rows(root, token_ids):
    """Read only official tied-head rows; never materialize a second full model."""
    import torch
    from safetensors import safe_open
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", config)
    if not text.get("tie_word_embeddings", False):
        raise ValueError("This profile requires an official tied output head")
    name = "model.language_model.embed_tokens.weight"
    index = json.loads((root / "model.safetensors.index.json").read_text())
    with safe_open(root / index["weight_map"][name], framework="pt", device="cpu") as source:
        view = source.get_slice(name)
        # Clone the tiny result to detach from the checkpoint's mapped storage.
        weight = torch.cat([view[i:i + 1, :] for i in token_ids]).clone()
    return weight, text.get("final_logit_softcapping")


class VllmDecisions(GemmaDecisions):
    def __init__(self, config):
        root = self._initialize(config)
        if config.get("quantization") != "fp8":
            raise ValueError("Only the measured FP8 profile is supported")
        import torch
        from vllm import LLM
        from vllm.config import PoolerConfig
        weight, self.softcap = checkpoint_rows(root, self.tokens)
        fraction = config["torch_memory_fraction"]
        if not 0 < fraction <= 0.5:
            raise ValueError("Torch allocator fraction must be at most one half")
        os.environ["GEMMA4_JEV_TORCH_FRACTION"] = str(fraction)
        self.llm = LLM(
            model=str(root), runner="pooling", convert="embed",
            hf_overrides={"architectures": ["Gemma4ForCausalLM"]},
            pooler_config=PoolerConfig(task="embed", seq_pooling_type="LAST", use_activation=False),
            dtype="bfloat16", quantization="fp8", load_format="runai_streamer",
            model_loader_extra_config={"concurrency": config["threads"],
                                       "memory_limit": config["loader_buffer_gib"] * 1024**3},
            max_model_len=config["max_input_tokens"], max_num_seqs=config["batch_size"],
            max_num_batched_tokens=config["max_input_tokens"],
            kv_cache_memory_bytes=config["kv_cache_gib"] * 1024**3,
            gpu_memory_utilization=0.5, enforce_eager=True,
            enable_prefix_caching=False, enable_chunked_prefill=False,
            trust_remote_code=False, seed=20261028,
            worker_cls="bounded_worker.BoundedWorker")
        self.device = torch.device("cuda:0")
        self.readout_weight = weight.to(self.device, dtype=torch.bfloat16)
        self.runtime_metadata = {"engine": "vllm_raw_last_hidden_state",
                                 "engine_version": importlib.metadata.version("vllm"),
                                 "quantization": "fp8_online_from_official_bf16"}

    def _score_batch(self, prompts):
        requests = [{"prompt_token_ids": self.tokenizer.encode(prompt, add_special_tokens=False)}
                    for prompt in prompts]
        outputs = self.llm.embed(requests, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise RuntimeError("Pooling runtime omitted an input")
        hidden = self.torch.tensor([output.outputs.embedding for output in outputs],
                                   device=self.device, dtype=self.torch.bfloat16)
        return selected_logits(hidden, self.readout_weight, range(len(self.tokens)),
                               self.softcap).cpu().tolist()
