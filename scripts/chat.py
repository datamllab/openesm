#!/usr/bin/env python3
"""Interactive terminal for ESM checkpoints.

Examples:
    python -m scripts.chat
    python -m scripts.chat --show-mcmc
    python -m scripts.chat --show-mcmc --verbose
    python -m scripts.chat -c /path/to/checkpoint
"""

import argparse
import sys
import os
import time
import torch
from typing import Optional, List, Dict, Any, Tuple


from esm.config import bootstrap_assets
from esm.generate import call_model_forward_decode, sample_top_p


for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
    if var in os.environ:
        del os.environ[var]


os.environ["ESM_OFFLINE_MODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


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
║   ███████╗██████╗ ████████╗     ██████╗██╗  ██╗ █████╗ ████████╗             ║
║   ██╔════╝██╔══██╗╚══██╔══╝    ██╔════╝██║  ██║██╔══██╗╚══██╔══╝             ║
║   █████╗  ██████╔╝   ██║       ██║     ███████║███████║   ██║                ║
║   ██╔══╝  ██╔══██╗   ██║       ██║     ██╔══██║██╔══██║   ██║                ║
║   ███████╗██████╔╝   ██║       ╚██████╗██║  ██║██║  ██║   ██║                ║
║   ╚══════╝╚═════╝    ╚═╝        ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝                ║
║                                                                              ║
║           Energy-Based Transformer Interactive Chat Terminal                 ║
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
        embed_dim = getattr(
            self.hparams, "embedding_dim", getattr(self.hparams, "dim", "unknown")
        )
        n_layers = getattr(
            self.hparams, "num_layers", getattr(self.hparams, "n_layers", "unknown")
        )
        n_heads = getattr(
            self.hparams, "num_heads", getattr(self.hparams, "n_heads", "unknown")
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

        print()
        print(f"  {Colors.BOLD}Model configuration:{Colors.RESET}")
        print(f"    - Embedding dimension: {embed_dim}")
        print(f"    - Layers: {n_layers}")
        print(f"    - Attention heads: {n_heads}")
        print(f"    - MCMC steps: {mcmc_steps}")

        if isinstance(alpha_val, float):
            print(f"    - MCMC step size (alpha): {alpha_val:.6f}")
        else:
            print(f"    - MCMC step size (alpha): {alpha_val}")

        print(f"    - Context length: {ctx_len}")
        print()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        stop_tokens: Optional[List[int]] = None,
        stream: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text using the shared generation helpers."""

        inner_tok = getattr(self.tokenizer, "tokenizer", None)
        if inner_tok is not None and hasattr(inner_tok, "encode_special"):
            bos_id = inner_tok.get_bos_token_id()
            user_start = inner_tok.encode_special("<|user_start|>")
            user_end = inner_tok.encode_special("<|user_end|>")
            asst_start = inner_tok.encode_special("<|assistant_start|>")
            content_ids = inner_tok.encode(prompt)
            prompt_tokens_list = (
                [bos_id, user_start] + content_ids + [user_end, asst_start]
            )
        else:
            encoded = self.tokenizer.encode(prompt)
            prompt_tokens_list = (
                encoded if isinstance(encoded, list) else encoded.tolist()
            )
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_id is not None and (
                not prompt_tokens_list or prompt_tokens_list[0] != bos_id
            ):
                prompt_tokens_list = [bos_id] + prompt_tokens_list

        if (
            hasattr(self.tokenizer, "bos_token_id")
            and self.tokenizer.bos_token_id is not None
        ):
            pad_id = self.tokenizer.bos_token_id
        elif (
            hasattr(self.tokenizer, "eos_token_id")
            and self.tokenizer.eos_token_id is not None
        ):
            pad_id = self.tokenizer.eos_token_id
        else:
            pad_id = 0

        bsz = 1
        min_prompt_len = len(prompt_tokens_list)
        max_prompt_len = len(prompt_tokens_list)

        ctx_len = getattr(
            self.hparams, "context_length", getattr(self.hparams, "max_seq_len", 2048)
        )
        total_len = min(ctx_len, max_tokens + max_prompt_len)

        tokens = torch.full(
            (bsz, total_len), pad_id, dtype=torch.long, device=self.device
        )
        tokens[0, : len(prompt_tokens_list)] = torch.tensor(
            prompt_tokens_list, dtype=torch.long, device=self.device
        )

        input_text_mask = torch.zeros(
            bsz, total_len, dtype=torch.bool, device=self.device
        )
        input_text_mask[0, : len(prompt_tokens_list)] = True

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.device)
        start_time = time.time()

        stop_token_ids = set()
        inner_tok = getattr(self.tokenizer, "tokenizer", None)
        if inner_tok is not None and hasattr(inner_tok, "encode_special"):
            asst_end_id = inner_tok.encode_special("<|assistant_end|>")
            if asst_end_id is not None:
                stop_token_ids.add(asst_end_id)

        if not stop_token_ids:
            stop_token_ids.add(pad_id)

        with torch.no_grad():
            if min_prompt_len == total_len:
                logits = call_model_forward_decode(
                    self.hparams, self.model, tokens, prev_pos, bsz
                )

            for cur_pos in range(min_prompt_len, total_len):
                input_tokens = tokens[:, :cur_pos]

                logits = call_model_forward_decode(
                    self.hparams, self.model, input_tokens, prev_pos, bsz
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

                if stream and cur_pos >= min_prompt_len:
                    token_text = self.tokenizer.decode(
                        [next_token.item()], skip_special_tokens=True
                    )
                    print(token_text, end="", flush=True)

                is_stop = torch.zeros(bsz, dtype=torch.bool, device=self.device)
                for sid in stop_token_ids:
                    is_stop |= next_token == sid
                eos_reached |= (~input_text_mask[:, cur_pos]) & is_stop
                prev_pos = cur_pos

                if all(eos_reached):
                    break

        toks = tokens[0].tolist()
        start = len(prompt_tokens_list)
        toks = toks[start : len(prompt_tokens_list) + max_tokens]

        for sid in stop_token_ids:
            if sid in toks:
                toks = toks[: toks.index(sid)]

        generated_text = self.tokenizer.decode(toks, skip_special_tokens=True)

        stats = {
            "tokens_generated": len(toks),
            "total_time": time.time() - start_time,
            "tokens_per_second": len(toks) / (time.time() - start_time)
            if (time.time() - start_time) > 0
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

    args = parser.parse_args()

    print_banner()

    if not os.path.exists(args.checkpoint):
        print_colored(f"Error: checkpoint not found: {args.checkpoint}", Colors.RED)
        return 1

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

    while True:
        try:
            user_input = input(f"{Colors.BLUE}You:{Colors.RESET} ")

            cmd = user_input.strip().lower()

            if cmd in ["/quit", "/exit"]:
                print_colored("\nGoodbye!", Colors.GREEN)
                break

            if cmd == "/clear":
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
                generated_text, stats = engine.generate(
                    prompt=user_input,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                )
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
