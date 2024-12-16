class PrepareCycleInputsNotSet(Exception):
    """Exception raised when class has not properly set"""

    def __init__(self, msg: str | None = None):
        self.msg = (
            msg
            or "_prepare_cycle_inputs is not corrently set. If you are modifying CycleTrainer.__init__ "
            "make sure to call self.set_cycle_inputs_fn(), optionally with a valid function."
        )
        super().__init__(self.msg)
