import ast


def check_python(s, unused_args):
    try:
        ast.parse(s)
        return True
    except Exception:
        return False


def check_mnist(s, text_to_array):
    '''
        Checks if s is a valid image string.
    '''
    arr = text_to_array(s)
    return bool(arr.sum() != 0)


SEQ_CHECKS = {
    "python": check_python,
    "mnist": check_mnist,
    None: lambda x, y: False
}
