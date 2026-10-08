#!/usr/bin/env python3
"""Interactive terminal for ESM checkpoints.

Examples:
    python -m scripts.chat
    python -m scripts.chat --show-mcmc
    python -m scripts.chat --show-mcmc --verbose
    python -m scripts.chat -c /path/to/checkpoint
    python -m scripts.chat --web -c /path/to/checkpoint --port 8000
"""

import argparse
import asyncio
import json
import queue
import sys
import os
import time
import torch
from typing import Any, Dict, Iterable, List, Optional, Tuple


from esm.config import bootstrap_assets
from esm.modeling_esm import call_model_forward_decode, sample_top_p


for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
    if var in os.environ:
        del os.environ[var]


os.environ["ESM_OFFLINE_MODE"] = "1"


def _bootstrap_assets() -> None:
    """Use the configured tokenizer/assets unless the environment overrides it."""

    bootstrap_assets("train")


class Colors:
    BLUE = "\033[1;34m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    CYAN = "\033[1;36m"
    MAGENTA = "\033[1;35m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    MCMC_COLORS = [
        "\033[38;5;196m",
        "\033[38;5;208m",
        "\033[38;5;226m",
        "\033[38;5;118m",
        "\033[38;5;51m",
    ]


def print_colored(text: str, color: str):
    """Print text with ANSI color codes."""
    print(f"{color}{text}{Colors.RESET}")


def print_banner():
    """Print the startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║      ███████╗ ██████╗ ███╗   ███╗     ██████╗ ██╗  ██╗ █████╗ ████████╗   ║
║      ██╔════╝██╔════╝ ████╗ ████║    ██╔════╝ ██║  ██║██╔══██╗╚══██╔══╝   ║
║      █████╗  ╚█████╗  ██╔████╔██║    ██║      ███████║███████║   ██║      ║
║      ██╔══╝   ╚═══██╗ ██║╚██╔╝██║    ██║      ██╔══██║██╔══██║   ██║      ║
║      ███████╗██████╔╝ ██║ ╚═╝ ██║    ╚██████╗ ██║  ██║██║  ██║   ██║      ║
║      ╚══════╝╚═════╝  ╚═╝     ╚═╝     ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝      ║
║                                                                              ║
║                         ESM Chat Terminal                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print_colored(banner, Colors.CYAN)


class ESMChatEngine:
    """Interactive generation engine for an ESM checkpoint."""

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        show_mcmc: bool = False,
        verbose: bool = False,
        show_energy: bool = False,
        show_distribution: bool = False,
    ):
        self.device = device
        self.tokenizer_path = tokenizer_path
        self.dtype = dtype
        self.show_mcmc = show_mcmc
        self.verbose = verbose
        self.show_energy = show_energy
        self.show_distribution = show_distribution

        self.model = None
        self.tokenizer = None
        self.hparams = None

        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """Load the model and tokenizer."""
        print_colored("Loading model...", Colors.YELLOW)

        from esm.modeling_esm import load_checkpoint

        wrapper, tokenizer, hparams, resolved_device = load_checkpoint(
            checkpoint_path,
            device=self.device,
            dtype=self.dtype,
            tokenizer_path=self.tokenizer_path,
        )
        self.device = resolved_device
        self.model = wrapper.model
        self.tokenizer = tokenizer

        class HParamsNamespace:
            def __init__(self, values):
                for key, value in values.items():
                    setattr(self, key, value)

        self.hparams = (
            HParamsNamespace(hparams) if isinstance(hparams, dict) else hparams
        )
        print_colored("✓ Model loaded and initialized", Colors.GREEN)
        self._print_model_info()

    def _print_model_info(self):
        """Print model metadata."""
        info = self.get_model_info()

        print()
        print(f"  {Colors.BOLD}Model configuration:{Colors.RESET}")
        print(f"    - Embedding dimension: {info['embedding_dim']}")
        print(f"    - Layers: {info['num_layers']}")
        print(f"    - Attention heads: {info['num_heads']}")
        print(f"    - MCMC steps: {info['mcmc_steps']}")
        print(f"    - MCMC step size (alpha): {info['alpha']}")
        print(f"    - Context length: {info['context_length']}")
        print()

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for the terminal and web interfaces."""

        embed_dim = getattr(
            self.hparams, "embedding_dim", getattr(self.hparams, "dim", "unknown")
        )
        n_layers = getattr(
            self.hparams,
            "num_layers",
            getattr(
                self.hparams,
                "num_transformer_blocks",
                getattr(self.hparams, "n_layers", "unknown"),
            ),
        )
        n_heads = getattr(
            self.hparams,
            "num_heads",
            getattr(
                self.hparams,
                "multiheaded_attention_heads",
                getattr(self.hparams, "n_heads", "unknown"),
            ),
        )
        mcmc_steps = getattr(self.hparams, "mcmc_num_steps", "unknown")
        ctx_len = getattr(
            self.hparams, "context_length", getattr(self.hparams, "max_seq_len", "unknown")
        )

        alpha_val = "unknown"
        if hasattr(self.model, "alpha"):
            alpha_val = (
                self.model.alpha.item()
                if isinstance(self.model.alpha, torch.Tensor)
                else self.model.alpha
            )

        if isinstance(alpha_val, (float, int)):
            alpha_val = float(alpha_val)
        return {
            "embedding_dim": embed_dim,
            "num_layers": n_layers,
            "num_heads": n_heads,
            "mcmc_steps": mcmc_steps,
            "alpha": alpha_val,
            "context_length": ctx_len,
            "show_mcmc": self.show_mcmc,
            "verbose": self.verbose,
            "show_energy": self.show_energy,
            "show_distribution": self.show_distribution,
        }

    def _tokenizer_core(self):
        return getattr(self.tokenizer, "tokenizer", self.tokenizer)

    def _decode(self, token_ids: Iterable[int]) -> str:
        try:
            return self.tokenizer.decode(
                list(token_ids), skip_special_tokens=True
            )
        except TypeError:
            return self.tokenizer.decode(list(token_ids))

    def _build_prompt_tokens(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[int]:
        if messages is None:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            messages = [{"role": "user", "content": prompt}]
        if not messages:
            raise ValueError("messages must contain at least one message")

        tokenizer = self._tokenizer_core()
        if hasattr(tokenizer, "encode_special"):
            prompt_tokens = [tokenizer.get_bos_token_id()]
            for message in messages:
                role = message.get("role")
                content = message.get("content", "")
                if role == "user":
                    prompt_tokens.extend(
                        [
                            tokenizer.encode_special("<|user_start|>"),
                            *tokenizer.encode(content),
                            tokenizer.encode_special("<|user_end|>"),
                        ]
                    )
                elif role == "assistant":
                    prompt_tokens.extend(
                        [
                            tokenizer.encode_special("<|assistant_start|>"),
                            *tokenizer.encode(content),
                            tokenizer.encode_special("<|assistant_end|>"),
                        ]
                    )
                else:
                    raise ValueError(f"Unsupported chat role: {role!r}")
            if messages[-1].get("role") != "user":
                raise ValueError("The final chat message must be from the user")
            prompt_tokens.append(tokenizer.encode_special("<|assistant_start|>"))
            return prompt_tokens

        last_user = next(
            (message.get("content", "") for message in reversed(messages)
             if message.get("role") == "user"),
            "",
        )
        encoded = self.tokenizer.encode(last_user)
        prompt_tokens = encoded if isinstance(encoded, list) else encoded.tolist()
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_id is not None and (not prompt_tokens or prompt_tokens[0] != bos_id):
            prompt_tokens.insert(0, bos_id)
        return prompt_tokens

    def generate_stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        stop_tokens: Optional[List[int]] = None,
    ):
        """Yield generated text for a prompt or a complete chat history."""

        prompt_tokens = self._build_prompt_tokens(prompt, messages)
        pad_id = getattr(self.tokenizer, "bos_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)

        context_length = int(
            getattr(self.hparams, "context_length", getattr(self.hparams, "max_seq_len", 2048))
        )
        if len(prompt_tokens) >= context_length:
            raise ValueError(
                f"Chat history has {len(prompt_tokens)} tokens, but the model "
                f"context length is {context_length}."
            )
        max_tokens = min(max(int(max_tokens), 1), context_length - len(prompt_tokens))

        tokens = torch.full(
            (1, len(prompt_tokens) + max_tokens),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )
        tokens[0, : len(prompt_tokens)] = torch.tensor(
            prompt_tokens, dtype=torch.long, device=self.device
        )
        input_text_mask = torch.zeros_like(tokens, dtype=torch.bool)
        input_text_mask[0, : len(prompt_tokens)] = True

        tokenizer = self._tokenizer_core()
        stop_token_ids = set(stop_tokens or [])
        if hasattr(tokenizer, "encode_special"):
            stop_token_ids.add(tokenizer.encode_special("<|assistant_end|>"))
        if not stop_token_ids:
            stop_token_ids.add(pad_id)

        eos_reached = False
        with torch.no_grad():
            for cur_pos in range(len(prompt_tokens), tokens.shape[1]):
                logits = call_model_forward_decode(
                    self.hparams, self.model, tokens[:, :cur_pos], start_pos=0, bsz=1
                )
                if temperature > 0:
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1)
                next_token = next_token.reshape(-1)
                next_token = torch.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                )
                tokens[:, cur_pos] = next_token
                token_id = int(next_token.item())
                if token_id in stop_token_ids:
                    eos_reached = True
                else:
                    token_text = self._decode([token_id])
                    if token_text:
                        yield token_text
                if eos_reached:
                    break

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        stop_tokens: Optional[List[int]] = None,
        stream: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text from one prompt or a multi-turn chat history."""

        start_time = time.time()
        pieces = []
        for token_text in self.generate_stream(
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_tokens=stop_tokens,
        ):
            pieces.append(token_text)
            if stream:
                print(token_text, end="", flush=True)

        generated_text = "".join(pieces)
        elapsed = time.time() - start_time

        stats = {
            "tokens_generated": len(self._tokenizer_core().encode(generated_text)),
            "total_time": elapsed,
            "tokens_per_second": len(self._tokenizer_core().encode(generated_text)) / elapsed
            if elapsed > 0
            else 0,
            "avg_energy_change": 0,
            "avg_token_prob": 0,
        }

        return generated_text, stats


def print_help():
    """Print available commands."""
    print()
    print(f"{Colors.BOLD}Available commands:{Colors.RESET}")
    print("  /quit, /exit        - Exit the session")
    print("  /clear              - Clear the conversation")
    print("  /temp <value>       - Set temperature (0.0-2.0)")
    print("  /topp <value>       - Set top-p (0.0-1.0)")
    print("  /tokens <value>     - Set the maximum token count (1-4096)")
    print("  /mcmc               - Toggle MCMC display")
    print("  /verbose            - Toggle verbose mode")
    print("  /energy             - Toggle energy display")
    print("  /status             - Show current settings")
    print("  /info               - Show model information")
    print("  /help               - Show this help")
    print()


def print_status(
    engine: ESMChatEngine, temperature: float, top_p: float, max_tokens: int
):
    """Print the current generation settings."""
    print()
    print(f"{Colors.BOLD}Current settings:{Colors.RESET}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-P: {top_p}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Show MCMC: {'yes' if engine.show_mcmc else 'no'}")
    print(f"  Verbose mode: {'yes' if engine.verbose else 'no'}")
    print(f"  Show energy: {'yes' if engine.show_energy else 'no'}")
    print()


def print_generation_stats(stats: Dict[str, Any]):
    """Print generation statistics."""
    print()
    print(f"{Colors.GRAY}────────────────────────────────────────{Colors.RESET}")
    print(f"{Colors.BOLD}Generation statistics:{Colors.RESET}")
    print(f"  Generated tokens: {stats['tokens_generated']}")
    print(f"  Total time: {stats['total_time']:.2f}s")
    print(f"  Speed: {stats['tokens_per_second']:.2f} tokens/s")
    print(f"  Average token probability: {stats['avg_token_prob']:.4f}")
    print(f"  Average energy change: {stats['avg_energy_change']:.4f}")
    print(f"{Colors.GRAY}────────────────────────────────────────{Colors.RESET}")
    print()


def _handle_web_command(
    command: str,
    engine: ESMChatEngine,
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a terminal-style command and return a JSON-safe response."""

    parts = command.strip().split()
    if not parts:
        return {"ok": False, "message": "Empty command"}

    name = parts[0].lower()
    if name == "/temperature":
        name = "/temp"
    if name in {"/help", "help"}:
        return {"result": (
            "Available commands:\n"
            "  /temp [value]     - Show or set temperature (0.0-2.0)\n"
            "  /topp [value]     - Show or set Top-P (0.0-1.0)\n"
            "  /tokens [value]   - Show or set maximum tokens (1-4096)\n"
            "  /mcmc             - Toggle MCMC display\n"
            "  /verbose          - Toggle verbose mode\n"
            "  /energy           - Toggle energy display\n"
            "  /status           - Show current settings\n"
            "  /info             - Show model information\n"
            "  /clear            - Clear conversation history\n"
            "  /help             - Show this help"
        )}
    if name == "/clear":
        return {"result": "Conversation cleared.", "action": "clear"}
    if name in {"/quit", "/exit"}:
        return {"result": "Close this page to exit web chat."}
    if name in {"/status", "status"}:
        info = engine.get_model_info()
        return {"result": (
            f"Current settings:\n  Temperature: {runtime['temperature']}\n"
            f"  Top-P: {runtime['top_p']}\n"
            f"  Maximum tokens: {runtime['max_tokens']}\n"
            f"  Show MCMC: {'yes' if info['show_mcmc'] else 'no'}\n"
            f"  Verbose mode: {'yes' if info['verbose'] else 'no'}\n"
            f"  Show energy: {'yes' if info['show_energy'] else 'no'}"
        )}
    if name in {"/info", "info"}:
        info = engine.get_model_info()
        return {"result": (
            f"Model configuration:\n  Embedding dimension: {info['embedding_dim']}\n"
            f"  Layers: {info['num_layers']}\n  Attention heads: {info['num_heads']}\n"
            f"  MCMC steps: {info['mcmc_steps']}\n"
            f"  MCMC step size (alpha): {info['alpha']}\n"
            f"  Context length: {info['context_length']}"
        )}
    if name in {"/mcmc", "mcmc"}:
        engine.show_mcmc = not engine.show_mcmc
        return {"result": f"✓ MCMC display {'enabled' if engine.show_mcmc else 'disabled'}"}
    if name in {"/verbose", "verbose"}:
        engine.verbose = not engine.verbose
        return {"result": f"✓ Verbose mode {'enabled' if engine.verbose else 'disabled'}"}
    if name in {"/energy", "energy"}:
        engine.show_energy = not engine.show_energy
        return {"result": f"✓ Energy display {'enabled' if engine.show_energy else 'disabled'}"}

    if name in {"/temp", "temp"}:
        if len(parts) < 2:
            return {"result": f"Current temperature: {runtime['temperature']}"}
        try:
            value = float(parts[1])
        except ValueError:
            return {"result": "✗ Invalid temperature value", "error": True}
        if not 0 <= value <= 2:
            return {"result": "✗ Temperature must be between 0.0 and 2.0", "error": True}
        runtime["temperature"] = value
        return {"result": f"✓ Temperature set to {value}"}

    if name in {"/topp", "topp"}:
        if len(parts) < 2:
            return {"result": f"Current Top-P: {runtime['top_p']}"}
        try:
            value = float(parts[1])
        except ValueError:
            return {"result": "✗ Invalid Top-P value", "error": True}
        if not 0 <= value <= 1:
            return {"result": "✗ Top-P must be between 0.0 and 1.0", "error": True}
        runtime["top_p"] = value
        return {"result": f"✓ Top-P set to {value}"}

    if name in {"/tokens", "tokens"}:
        if len(parts) < 2:
            return {"result": f"Current maximum tokens: {runtime['max_tokens']}"}
        try:
            value = int(parts[1])
        except ValueError:
            return {"result": "✗ Invalid token limit", "error": True}
        if not 1 <= value <= 4096:
            return {"result": "✗ Maximum tokens must be between 1 and 4096", "error": True}
        runtime["max_tokens"] = value
        return {"result": f"✓ Maximum tokens set to {value}"}

    return {"result": f"Unknown command: {name}. Enter /help to list available commands.", "error": True}


def run_web(
    engine: ESMChatEngine,
    host: str,
    port: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> None:
    """Run the browser UI and streaming chat endpoint."""

    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, StreamingResponse
        from pydantic import BaseModel
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "Web chat requires the optional dependencies. "
            "Install them with `uv sync --extra web`."
        ) from exc

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        messages: List[ChatMessage]
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        max_tokens: Optional[int] = None

    class CommandRequest(BaseModel):
        command: str

    app = FastAPI(title="ESM Chat")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    runtime = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    generation_lock = asyncio.Lock()
    page = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <title>ESM Chat</title>
    <style>
        :root { color-scheme: light; }
        * { box-sizing: border-box; }
        html, body { height: 100%; margin: 0; }
        body {
            font-family: ui-sans-serif, -apple-system, system-ui, "Segoe UI", Helvetica, Arial, sans-serif;
            background-color: #ffffff; color: #111827;
            min-height: 100dvh; display: flex; flex-direction: column;
        }
        .header { background-color: #ffffff; padding: 1rem 1.5rem; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #f3f4f6; }
        .header-left { display: flex; align-items: center; gap: 0.75rem; }
        .header h1 { font-size: 1.25rem; font-weight: 600; margin: 0; color: #111827; }
        .header-tag { font-size: 0.7rem; background: #eef2ff; color: #4f46e5; padding: 0.15rem 0.5rem; border-radius: 0.25rem; font-weight: 500; }
        .new-btn { width: 32px; height: 32px; padding: 0; border: 1px solid #e5e7eb; border-radius: 0.5rem; background: #fff; color: #6b7280; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; }
        .new-btn:hover { background: #f3f4f6; border-color: #d1d5db; color: #374151; }
        .chat-container { flex: 1; overflow-y: auto; background: #ffffff; }
        .chat-wrapper { max-width: 48rem; margin: 0 auto; padding: 2rem 1.5rem 3rem; display: flex; flex-direction: column; gap: 0.75rem; }
        .message { display: flex; margin-bottom: 0.5rem; color: #0d0d0d; }
        .message.assistant { justify-content: flex-start; }
        .message.user { justify-content: flex-end; }
        .message-content { white-space: pre-wrap; line-height: 1.6; max-width: 100%; }
        .message.assistant .message-content { background: transparent; border: none; cursor: pointer; border-radius: 0.5rem; padding: 0.5rem; margin-left: -0.5rem; transition: background-color 0.2s; }
        .message.assistant .message-content:hover { background: #f9fafb; }
        .message.user .message-content { background-color: #f3f4f6; border-radius: 1.25rem; padding: 0.8rem 1rem; max-width: 65%; cursor: pointer; transition: background-color 0.2s; }
        .message.user .message-content:hover { background-color: #e5e7eb; }
        .message.console .message-content { font-family: 'Monaco','Menlo','Consolas','Courier New', monospace; font-size: 0.85rem; background: #f8fafc; border: 1px solid #e2e8f0; padding: 0.75rem 1rem; color: #374151; max-width: 85%; border-radius: 0.5rem; }
        .input-container { background: #fff; padding: 1rem; padding-bottom: calc(1rem + env(safe-area-inset-bottom)); }
        .input-wrapper { max-width: 48rem; margin: 0 auto; display: flex; gap: 0.75rem; align-items: flex-end; }
        .chat-input { flex: 1; padding: 0.8rem 1rem; border: 1px solid #d1d5db; border-radius: 0.75rem; background: #fff; color: #111827; font-size: 1rem; line-height: 1.5; resize: none; outline: none; min-height: 54px; max-height: 200px; transition: border-color 0.2s, box-shadow 0.2s; }
        .chat-input::placeholder { color: #9ca3af; }
        .chat-input:focus { border-color: #4f46e5; box-shadow: 0 0 0 3px rgba(79,70,229,0.1); }
        .send-btn { flex-shrink: 0; width: 54px; height: 54px; border: 1px solid #111827; border-radius: 0.75rem; background: #111827; color: #fff; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: background 0.2s, border-color 0.2s; }
        .send-btn:hover:not(:disabled) { background: #4f46e5; border-color: #4f46e5; }
        .send-btn:disabled { cursor: not-allowed; border-color: #d1d5db; background: #e5e7eb; color: #9ca3af; }
        .typing-indicator { display: inline-block; color: #6b7280; letter-spacing: 0.15em; }
        .typing-indicator::after { content: '···'; animation: typing 1.4s infinite; }
        @keyframes typing { 0%,60%,100%{opacity:.2;} 30%{opacity:1;} }
        .error-message { background: #fee2e2; border: 1px solid #fecaca; color: #b91c1c; padding: 0.75rem 1rem; border-radius: 0.75rem; margin-top: 0.5rem; }
    </style>
</head>
<body>
    <div class="header"><div class="header-left">
        <button class="new-btn" onclick="newConversation()" title="New Conversation (Ctrl+Shift+N)">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"/><path d="M5 12h14"/></svg>
        </button>
        <h1>ESM Chat</h1><span class="header-tag">Energy-Steered Model</span>
    </div></div>
    <div class="chat-container" id="chatContainer"><div class="chat-wrapper" id="chatWrapper"></div></div>
    <div class="input-container"><div class="input-wrapper">
        <textarea id="chatInput" class="chat-input" placeholder="Type a message or enter /help for commands..." rows="1" onkeydown="handleKeyDown(event)"></textarea>
        <button id="sendButton" class="send-btn" onclick="sendMessage()" disabled>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg>
        </button>
    </div></div>
<script>
const API_URL = '';
const chatContainer = document.getElementById('chatContainer');
const chatWrapper = document.getElementById('chatWrapper');
const chatInput = document.getElementById('chatInput');
const sendButton = document.getElementById('sendButton');
let messages = [];
let isGenerating = false;

chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    sendButton.disabled = !this.value.trim() || isGenerating;
});

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.shiftKey && e.key === 'N') { e.preventDefault(); if (!isGenerating) newConversation(); }
});

function newConversation() {
    messages = []; chatWrapper.innerHTML = '';
    chatInput.value = ''; chatInput.style.height = 'auto';
    sendButton.disabled = false; isGenerating = false; chatInput.focus();
}

function addMessage(role, content, messageIndex) {
    const div = document.createElement('div'); div.className = 'message ' + role;
    const c = document.createElement('div'); c.className = 'message-content'; c.textContent = content;
    if (role === 'user' && messageIndex !== undefined) {
        c.title = 'Click to edit and restart from here';
        c.addEventListener('click', () => { if (!isGenerating) editMessage(messageIndex); });
    }
    if (role === 'assistant' && messageIndex !== undefined) {
        c.title = 'Click to regenerate this response';
        c.addEventListener('click', () => { if (!isGenerating) regenerateMessage(messageIndex); });
    }
    div.appendChild(c); chatWrapper.appendChild(div); chatContainer.scrollTop = chatContainer.scrollHeight;
    return c;
}

function editMessage(idx) {
    if (idx < 0 || idx >= messages.length || messages[idx].role !== 'user') return;
    chatInput.value = messages[idx].content; chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
    messages = messages.slice(0, idx);
    const all = chatWrapper.querySelectorAll('.message');
    for (let i = idx; i < all.length; i++) all[i].remove();
    sendButton.disabled = false; chatInput.focus();
}

async function regenerateMessage(idx) {
    if (idx < 0 || idx >= messages.length || messages[idx].role !== 'assistant') return;
    messages = messages.slice(0, idx);
    const all = chatWrapper.querySelectorAll('.message');
    for (let i = idx; i < all.length; i++) all[i].remove();
    await generateAssistantResponse();
}

async function generateAssistantResponse() {
    isGenerating = true; sendButton.disabled = true;
    const el = addMessage('assistant', ''); el.innerHTML = '<span class="typing-indicator"></span>';
    try {
        const resp = await fetch(API_URL + '/chat/completions', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({messages: messages})
        });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const reader = resp.body.getReader(); const dec = new TextDecoder(); let full = ''; let pending = ''; el.textContent = '';
        while (true) {
            const {done, value} = await reader.read(); if (done) break;
            pending += dec.decode(value, {stream: true});
            const lines = pending.split('\n'); pending = lines.pop();
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const d = JSON.parse(line.slice(6));
                    if (d.token) { full += d.token; el.textContent = full; chatContainer.scrollTop = chatContainer.scrollHeight; }
                    if (d.error) { el.innerHTML = '<div class="error-message">Error: ' + d.error + '</div>'; }
                } catch (_) {}
            }
        }
        const aidx = messages.length; messages.push({role: 'assistant', content: full});
        el.title = 'Click to regenerate this response';
        el.addEventListener('click', () => { if (!isGenerating) regenerateMessage(aidx); });
    } catch (err) {
        el.innerHTML = '<div class="error-message">Error: ' + err.message + '</div>';
    } finally {
        isGenerating = false; sendButton.disabled = !chatInput.value.trim();
    }
}

async function handleSlashCommand(cmd) {
    try {
        const resp = await fetch(API_URL + '/command', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({command: cmd})
        });
        const data = await resp.json();
        if (data.action === 'clear') { newConversation(); return; }
        addMessage('console', data.result);
    } catch (err) { addMessage('console', 'Error: ' + err.message); }
}

async function sendMessage() {
    const msg = chatInput.value.trim(); if (!msg || isGenerating) return;
    chatInput.value = ''; chatInput.style.height = 'auto';
    if (msg.startsWith('/')) { await handleSlashCommand(msg); return; }
    const uidx = messages.length; messages.push({role: 'user', content: msg});
    addMessage('user', msg, uidx); await generateAssistantResponse();
}

sendButton.disabled = false; chatInput.focus();
fetch(API_URL + '/health').then(r => r.json()).then(d => {
    console.log('ESM Engine status:', d);
}).catch(() => {
    chatWrapper.innerHTML = '<div class="error-message">The ESM engine is not ready. Wait for the model to load, then refresh the page.</div>';
});
</script>
</body>
</html>"""

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return page

    @app.get("/health")
    async def health():
        return {"status": "ok", "ready": engine.model is not None, "device": str(engine.device)}

    @app.get("/status")
    async def status():
        info = engine.get_model_info()
        info.update(runtime)
        return info

    @app.post("/command")
    async def command(request: CommandRequest):
        return _handle_web_command(request.command, engine, runtime)

    @app.post("/chat/completions")
    async def completions(request: ChatRequest):
        messages = [
            message.model_dump() if hasattr(message, "model_dump") else message.dict()
            for message in request.messages
        ]
        if not messages:
            from fastapi import HTTPException

            raise HTTPException(status_code=400, detail="At least one message is required")
        if messages[-1].get("role") != "user":
            from fastapi import HTTPException

            raise HTTPException(status_code=400, detail="The final message must be from the user")
        temperature_value = (
            runtime["temperature"] if request.temperature is None else request.temperature
        )
        top_p_value = runtime["top_p"] if request.top_p is None else request.top_p
        max_tokens_value = (
            runtime["max_tokens"] if request.max_tokens is None else request.max_tokens
        )
        temperature_value = max(0.0, min(2.0, temperature_value))
        top_p_value = max(0.0, min(1.0, top_p_value))
        max_tokens_value = max(1, min(4096, max_tokens_value))

        async def stream_response():
            async with generation_lock:
                token_queue: queue.Queue = queue.Queue()

                def worker():
                    try:
                        for token in engine.generate_stream(
                            messages=messages,
                            max_tokens=max_tokens_value,
                            temperature=temperature_value,
                            top_p=top_p_value,
                        ):
                            token_queue.put({"token": token})
                    except Exception as exc:
                        import logging

                        logging.getLogger(__name__).exception("ESM generation failed")
                        token_queue.put({"error": str(exc)})
                    finally:
                        token_queue.put(None)

                task = asyncio.create_task(asyncio.to_thread(worker))
                while True:
                    item = await asyncio.to_thread(token_queue.get)
                    if item is None:
                        break
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                await task
                yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    print_colored(f"Web chat listening on http://{host}:{port}", Colors.GREEN)
    uvicorn.run(app, host=host, port=port)


def main():
    _bootstrap_assets()
    parser = argparse.ArgumentParser(description="ESM interactive chat")
    parser.add_argument(
        "-c", "--checkpoint", type=str, default="", help="Path to the ESM checkpoint"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Directory containing tokenizer.pkl and token_bytes.pt",
    )
    parser.add_argument(
        "-t", "--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p sampling threshold (default: 0.9)"
    )
    parser.add_argument(
        "-m", "--max-tokens", type=int, default=256, help="Maximum generated tokens (default: 256)"
    )
    parser.add_argument("--show-mcmc", action="store_true", help="Show MCMC steps")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--show-energy", action="store_true", help="Show energy changes")
    parser.add_argument(
        "--show-distribution", action="store_true", help="Show probability distribution changes"
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
        help="Data type (default: bfloat16)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument(
        "--web", action="store_true", help="Serve the browser chat UI instead of the terminal"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server host")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")

    args = parser.parse_args()

    print_banner()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    torch.set_float32_matmul_precision("medium")

    try:
        engine = ESMChatEngine(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            dtype=dtype,
            show_mcmc=args.show_mcmc,
            verbose=args.verbose,
            show_energy=args.show_energy,
            show_distribution=args.show_distribution,
        )
    except Exception as e:
        print_colored(f"Error: failed to load model - {str(e)}", Colors.RED)
        import traceback

        traceback.print_exc()
        return 1

    if args.web:
        try:
            run_web(
                engine,
                host=args.host,
                port=args.port,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:
            print_colored(f"Error: failed to start web chat - {exc}", Colors.RED)
            return 1
        return 0

    print("=" * 70)
    print_colored("ESM chat is ready!", Colors.GREEN)
    print("=" * 70)
    print("Instructions:")
    print("  - Enter text and press Enter to generate")
    print("  - Enter /quit or /exit to leave")
    print("  - Enter /help to list all commands")
    print("  - Enter /mcmc to toggle MCMC step display")
    print("=" * 70)
    print()

    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens

    session_total_tokens = 0
    session_total_time = 0.0
    session_turns = 0
    conversation: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input(f"{Colors.BLUE}You:{Colors.RESET} ")

            cmd = user_input.strip().lower()

            if cmd in ["/quit", "/exit"]:
                print_colored("\nGoodbye!", Colors.GREEN)
                break

            if cmd == "/clear":
                conversation.clear()
                os.system("clear" if os.name == "posix" else "cls")
                print_banner()
                continue

            if cmd == "/help":
                print_help()
                continue

            if cmd == "/status":
                print_status(engine, temperature, top_p, max_tokens)
                continue

            if cmd == "/info":
                engine._print_model_info()
                continue

            if cmd == "/mcmc":
                engine.show_mcmc = not engine.show_mcmc
                status = "on" if engine.show_mcmc else "off"
                print_colored(f"\n✓ MCMC display: {status}\n", Colors.GREEN)
                continue

            if cmd == "/verbose":
                engine.verbose = not engine.verbose
                status = "on" if engine.verbose else "off"
                print_colored(f"\n✓ Verbose mode: {status}\n", Colors.GREEN)
                continue

            if cmd == "/energy":
                engine.show_energy = not engine.show_energy
                status = "on" if engine.show_energy else "off"
                print_colored(f"\n✓ Energy display: {status}\n", Colors.GREEN)
                continue

            if cmd.startswith("/temp "):
                try:
                    new_temp = float(cmd.split()[1])
                    if 0.0 <= new_temp <= 2.0:
                        temperature = new_temp
                        print_colored(
                            f"\n✓ Temperature set to: {temperature}\n", Colors.GREEN
                        )
                    else:
                        print_colored("\n✗ Temperature must be between 0.0 and 2.0\n", Colors.RED)
                except (IndexError, ValueError):
                    print_colored("\n✗ Invalid temperature\n", Colors.RED)
                continue

            if cmd.startswith("/topp "):
                try:
                    new_topp = float(cmd.split()[1])
                    if 0.0 <= new_topp <= 1.0:
                        top_p = new_topp
                        print_colored(f"\n✓ Top-p set to: {top_p}\n", Colors.GREEN)
                    else:
                        print_colored("\n✗ Top-p must be between 0.0 and 1.0\n", Colors.RED)
                except (IndexError, ValueError):
                    print_colored("\n✗ Invalid top-p value\n", Colors.RED)
                continue

            if cmd.startswith("/tokens "):
                try:
                    new_tokens = int(cmd.split()[1])
                    if 1 <= new_tokens <= 4096:
                        max_tokens = new_tokens
                        print_colored(
                            f"\n✓ Maximum tokens set to: {max_tokens}\n", Colors.GREEN
                        )
                    else:
                        print_colored(
                            "\n✗ Maximum tokens must be between 1 and 4096\n", Colors.RED
                        )
                except (IndexError, ValueError):
                    print_colored("\n✗ Invalid token count\n", Colors.RED)
                continue

            if not user_input.strip():
                continue

            print(f"{Colors.GREEN}ESM:{Colors.RESET} ", end="", flush=True)

            try:
                conversation.append({"role": "user", "content": user_input})
                generated_text, stats = engine.generate(
                    messages=conversation,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                )
                conversation.append({"role": "assistant", "content": generated_text})
                print()

                session_total_tokens += stats["tokens_generated"]
                session_total_time += stats["total_time"]
                session_turns += 1

                if engine.verbose or engine.show_mcmc:
                    print_generation_stats(stats)
                else:
                    print(
                        f"{Colors.GRAY}  [{stats['tokens_generated']} tokens, {stats['total_time']:.1f}s, {stats['tokens_per_second']:.1f} tok/s]{Colors.RESET}"
                    )
                    print()

            except Exception as e:
                if conversation and conversation[-1].get("role") == "user":
                    conversation.pop()
                print()
                print_colored(f"✗ Generation error: {str(e)}", Colors.RED)
                import traceback

                traceback.print_exc()

        except KeyboardInterrupt:
            print_colored("\n\nCtrl+C received, exiting...", Colors.YELLOW)
            break
        except EOFError:
            print_colored("\n\nEOF received, exiting...", Colors.YELLOW)
            break

    if session_turns > 0:
        avg_tps = (
            session_total_tokens / session_total_time if session_total_time > 0 else 0
        )
        print()
        print("=" * 50)
        print("  Session summary")
        print("=" * 50)
        print(f"  Turns:             {session_turns}")
        print(f"  Generated tokens:  {session_total_tokens}")
        print(f"  Total time:        {session_total_time:.2f}s")
        print(f"  Average throughput: {avg_tps:.2f} tokens/s")
        print(
            f"  Average per turn:  {session_total_tokens / session_turns:.0f} tokens, {session_total_time / session_turns:.1f}s"
        )
        print("=" * 50)
        print(
            f"[SESSION_SUMMARY] turns={session_turns} tokens={session_total_tokens} time={session_total_time:.1f}s throughput={avg_tps:.2f}tok/s"
        )

    print_colored("\nSession ended.", Colors.GREEN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
