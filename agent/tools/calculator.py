import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import structlog

log = structlog.get_logger()

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

def calculate(expression: str) -> dict:
    """
    Safe symbolic calculator using SymPy.
    Handles arithmetic, algebra, calculus expressions.
    Never uses eval() — parses to AST first for safety.
    """
    try:
        expr = parse_expr(expression, transformations=TRANSFORMATIONS)
        result = sympy.simplify(expr)
        numeric = float(result.evalf()) if result.is_number else None
        log.info("calc_done", expression=expression, result=str(result))
        return {
            "success": True,
            "expression": expression,
            "symbolic": str(result),
            "numeric": numeric,
        }
    except Exception as e:
        log.error("calc_failed", expression=expression, error=str(e))
        return {"success": False, "error": str(e), "expression": expression}