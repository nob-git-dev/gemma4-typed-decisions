"""An original one-pass readout over the unmodified official Gemma backbone."""
import json
from pathlib import Path
import time

from decision_core import make_prompt, render_answer, validate_request
from order_ensemble import combine_orders, question_variants


def selected_logits(hidden, weight, token_ids, softcap=None, bias=None):
    import torch
    import torch.nn.functional as F
    indices = torch.as_tensor(token_ids, device=weight.device, dtype=torch.long)
    selected_bias = None if bias is None else bias.index_select(0, indices)
    logits = F.linear(hidden, weight.index_select(0, indices), selected_bias).float()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return logits


class GemmaDecisions:
    def __init__(self, config):
        root = self._initialize(config)
        from transformers import Gemma4ForConditionalGeneration
        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            root, dtype=self.torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="sdpa", local_files_only=True, trust_remote_code=False)
        self.model.eval()
        self.backbone = self.model.model.language_model
        self.head = self.model.get_output_embeddings()
        self.softcap = getattr(self.model.config.text_config, "final_logit_softcapping", None)
        self.device = self.head.weight.device

    def _initialize(self, config):
        import torch
        from transformers import AutoTokenizer
        self.config = config
        root = Path(config["model_path"])
        origin = json.loads((root / "origin.json").read_text())
        if origin["revision"] != config["revision"] or origin["model_id"] != config["model_id"]:
            raise RuntimeError("Checkpoint provenance does not match pinned configuration")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required for this profile")
        torch.set_num_threads(config["threads"])
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True,
                                                       padding_side="left", trust_remote_code=False)
        self.tokens = []
        for symbol in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            tokens = self.tokenizer.encode(symbol, add_special_tokens=False)
            if len(tokens) != 1:
                raise RuntimeError("Decision symbol is not one tokenizer token: " + symbol)
            self.tokens.append(tokens[0])
        if len(set(self.tokens)) != len(self.tokens):
            raise RuntimeError("Decision symbols do not have distinct token ids")
        return root

    def _score_batch(self, prompts):
        batch = self.tokenizer(prompts, add_special_tokens=False,
                               padding=True, return_tensors="pt").to(self.device)
        hidden = self.backbone(**batch, use_cache=False, return_dict=True).last_hidden_state[:, -1, :]
        return selected_logits(hidden, self.head.weight, self.tokens,
                               self.softcap, getattr(self.head, "bias", None)).cpu().tolist()

    def decide(self, body):
        torch = self.torch
        questions = validate_request(body, self.config["max_questions"])
        started = time.perf_counter()
        prompts, groups, compiled = [], [], list(questions.values())
        for q in compiled:
            variants = question_variants(q, self.config.get("option_orders", 1))
            groups.append((len(prompts), variants))
            for variant in variants:
                prompt = make_prompt(self.tokenizer, body["state"], variant)
                prefix_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                for symbol, token_id in zip(variant.symbols, self.tokens):
                    if self.tokenizer.encode(prompt + symbol, add_special_tokens=False) != prefix_ids + [token_id]:
                        raise ValueError("Decision token boundary is not stable")
                prompts.append(prompt)
        encoded = self.tokenizer(prompts, add_special_tokens=False)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        if max(lengths) > self.config["max_input_tokens"] or sum(lengths) > self.config["max_total_tokens"]:
            raise ValueError("Token limit exceeded; input was not truncated")
        answers, trace, logit_rows = {}, {}, []
        with torch.inference_mode():
            for start in range(0, len(prompts), self.config["batch_size"]):
                batch_prompts = prompts[start:start + self.config["batch_size"]]
                logit_rows.extend(self._score_batch(batch_prompts))
        if len(logit_rows) != len(prompts):
            raise RuntimeError("Backbone omitted a decision input")
        for key, q, (start, variants) in zip(questions, compiled, groups):
            rows = logit_rows[start:start + len(variants)]
            probs = combine_orders(q, variants, rows, self.config["temperature"],
                                   self.config.get("ensemble_method", "logit_mean"))
            answers[key] = render_answer(q, probs)
            if self.config.get("record_order_components", False):
                trace[key] = [{"names": list(variant.names), "logits": row[:len(variant.names)]}
                              for variant, row in zip(variants, rows)]
        torch.cuda.synchronize()
        metadata = {"base_model": self.config["model_id"], "revision": self.config["revision"],
                    "compute_ms": (time.perf_counter() - started) * 1000,
                    "calibration": "uncalibrated_candidate_softmax", "generated_tokens": 0,
                    "question_orders": {key: len(variants) for key, (_, variants) in zip(questions, groups)},
                    **getattr(self, "runtime_metadata", {})}
        if trace:
            metadata["order_components"] = trace
        return {"model": self.config["served_model"], "answers": answers,
                "usage": {"input_tokens": sum(lengths), "output_tokens": 0},
                "metadata": metadata}
