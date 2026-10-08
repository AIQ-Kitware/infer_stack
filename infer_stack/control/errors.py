"""Expected transport failures presented by the CLI without a traceback."""


class ControlError(ValueError):
    pass


class ControlTimeout(TimeoutError):
    pass
