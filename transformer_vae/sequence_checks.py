import ast


def check_python(s):
    try:
        ast.parse(s)
        return True
    except Exception:
        return False


def check_mnist(s):
    '''
    img = txt_to_img(s)
    '''
    pass


SEQ_CHECKS = {
    "python": check_python,
    "mnist": check_mnist,
    None: lambda x: False
}
