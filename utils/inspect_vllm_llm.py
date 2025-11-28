#!/usr/bin/env python3
"""
Inspect vLLM LLM object to see attributes and implementation details.
This helps confirm if 'llm_engine' is present and which LLM implementation is installed.
"""
import importlib
import inspect
import sys
import traceback


def main():
    try:
        import vllm
    except Exception as e:
        print("Failed to import vllm:", e)
        traceback.print_exc()
        return 2

    print("vllm version:", getattr(vllm, '__version__', 'unknown'))
    try:
        from vllm import LLM
        print("LLM class imported from vllm")
    except Exception as e:
        print("LLM import failed:", e)
        traceback.print_exc()
        return 3

    print("LLM class location:", inspect.getsourcefile(LLM))
    print("LLM class members (dir):")
    print('\n'.join([m for m in dir(LLM) if not m.startswith('_')]))

    # Instantiate with a tiny model (use gpt2) and inspect the instance attributes
    try:
        print("Attempting to instantiate LLM with small model 'gpt2' to avoid memory overhead")
        engine = LLM(model="gpt2", gpu_memory_utilization=0.2, tensor_parallel_size=1, pipeline_parallel_size=1)
        print("Instance dir (non-private members):")
        print('\n'.join([m for m in dir(engine) if not m.startswith('_')]))
        print("Has llm_engine attribute?:", hasattr(engine, 'llm_engine'))
        # Attempt to close gracefully
        try:
            engine.close()
        except Exception as e:
            print("Error closing engine:", e)
    except Exception as e:
        print("LLM instantiation failed (likely due to missing/unsupported architecture or memory):", e)
        traceback.print_exc()
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
