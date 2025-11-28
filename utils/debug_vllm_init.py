#!/usr/bin/env python3
"""
Simple vLLM engine init test. Use to verify vLLM can instantiate with a model and GPU memory setting.
"""
import argparse
import os


def run(model: str, gpu_mem: float, dtype: str, device: str = 'cuda', suppress_del_error: bool = False):
    try:
        # optional monkey-patch to avoid errors when __init__ fails and destructor expects attributes
        if suppress_del_error:
            import vllm
            if hasattr(vllm, 'LLM'):
                LLM_cls = vllm.LLM
                if not hasattr(LLM_cls, '__del__'):
                    pass
                else:
                    orig_del = LLM_cls.__del__
                    def safe_del(self):
                        if hasattr(self, 'llm_engine') and getattr(self, 'llm_engine', None):
                            try:
                                orig_del(self)
                            except Exception:
                                pass
                    LLM_cls.__del__ = safe_del
        from vllm import LLM
    except Exception as e:
        print("vLLM import failed:", e)
        return 2

    print("Attempting engine creation for:", model, "device:", device)
    if device.lower() == 'cpu':
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    try:
        engine_kwargs = dict(model=model, dtype=dtype, trust_remote_code=True, tensor_parallel_size=1, pipeline_parallel_size=1)
        if device.lower() == 'cpu':
            engine_kwargs['device'] = 'cpu'
        else:
            engine_kwargs['gpu_memory_utilization'] = gpu_mem
        engine = LLM(**engine_kwargs)
        print("Engine created OK")
        # Try to generate a small sample if possible
        try:
            out = engine.generate([{"prompt": "Hello", "max_tokens": 1}])
            print("Generation OK; sample out len:", len(list(out)))
        except Exception as e:
            print("Generation test failed (non-fatal):", e)
        # Shutdown gracefully if engine provides shutdown API
        try:
            if hasattr(engine, 'shutdown'):
                engine.shutdown()
            elif hasattr(engine, 'close'):
                engine.close()
            elif hasattr(engine, 'llm_engine') and hasattr(engine.llm_engine, 'shutdown'):
                engine.llm_engine.shutdown()
        except Exception as e:
            print('Failed to shutdown engine gracefully:', e)
    except Exception as e:
        print("Engine failed to create:", e)
        return 1
    return 0


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model name or local path")
    parser.add_argument("--gpu_mem", type=float, default=0.6)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda", choices=["cuda","cpu"], help="Device to attempt running on (cuda or cpu)")
    parser.add_argument("--suppress_del_error", action='store_true', help='Monkey-patch LLM.__del__ to avoid attribute error on failed init')
    args = parser.parse_args()
    return run(args.model, args.gpu_mem, args.dtype, device=args.device, suppress_del_error=args.suppress_del_error)


if __name__ == "__main__":
    raise SystemExit(cli())
