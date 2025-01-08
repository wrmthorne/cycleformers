class CycleModelError(Exception):
    """Exception raised when models are not properly configured for CycleTrainer.

    This exception is raised when models are either not provided or are invalid for use with CycleTrainer.
    """

    def __init__(self, message="CycleTrainer is missing valid models for training."):
        self.message = message
        super().__init__(self.message)


class MissingModelError(CycleModelError):
    """Exception raised when a model is not provided for CycleTrainer.

    This exception is raised when a model is not provided for CycleTrainer.
    """

    default_message = (
        "CycleTrainer requires two models to train but only received a single model. "
        "To train using adapter-switching, set `args.use_macct = True`. Otherwise, pass two separate models "
        "for cycle training."
    )

    def __init__(self, message=None):
        self.message = message or self.default_message
        super().__init__(self.message)


class MACCTModelError(Exception):
    """Exception raised when MACCT models are not properly configured for CycleTrainer.

    This exception is raised when MACCT models are either not provided or are invalid for use with CycleTrainer.
    """

    def __init__(self, message="There is something wrong with the MACCT model or configuration provided."):
        self.message = message
        super().__init__(self.message)


class InvalidCycleKeyError(Exception):
    """Exception raised when an invalid model key is provided to CycleTrainer.

    This exception is raised when a cycle key other than 'A' or 'B' is provided when configuring or accessing
    models in CycleTrainer.
    """

    def __init__(self, message="Invalid cycle key provided. Only 'A' and 'B' are valid keys for cycle training."):
        self.message = message
        super().__init__(self.message)
