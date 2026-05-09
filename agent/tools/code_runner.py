import sys
import io
import signal
import structlog

log = structlog.get_logger()

SAFE_BUILTINS = {
    "print": print, "range": range, "len": len, "sum": sum,
    "min": min, "max": max, "abs": abs, "round": round,
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed,
}

def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out after 5 seconds")

def run_code(code: str, timeout: int = 5) -> dict:
    """
    Sandboxed Python code execution.
    Restricts builtins to a safe whitelist — no file I/O, no imports, no os access.
    Enforces timeout via SIGALRM (Unix only).
    """
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture

    try:
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        exec(code, {"__builtins__": SAFE_BUILTINS})

        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

        output = stdout_capture.getvalue()
        log.info("code_runner_success", output_len=len(output))
        return {"success": True, "output": output, "error": None}

    except TimeoutError as e:
        return {"success": False, "output": "", "error": str(e)}
    except Exception as e:
        return {"success": False, "output": stdout_capture.getvalue(), "error": str(e)}
    finally:
        sys.stdout = old_stdout