def assertEqual(actual, expected, msg, first="Got", second="Expected"):
    if actual != expected:
        raise ValueError(msg + f' {first}: "{actual}" {second}: "{expected}"')


def assertIn(actual, expected, msg, first="Got", second="Expected one of"):
    if actual not in expected:
        raise ValueError(msg + f' {first}: "{actual}" {second}: {expected}')
