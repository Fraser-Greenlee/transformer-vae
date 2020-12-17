import ast


def check_python(s):
    try:
        ast.parse(s)
        return True
    except Exception:
        return False


SEQ_CHECKS = {"python": check_python}
