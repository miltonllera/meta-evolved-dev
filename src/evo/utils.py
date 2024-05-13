from typing import Literal, get_args
import evosax as ex
import equinox as eqx
from jaxtyping import PyTree


MemberName = Literal["mean", "best"]


class ModelFromState:
    def __init__(self, member_to_extract: MemberName = "mean"):
        if not member_to_extract in get_args(MemberName):
            raise RuntimeError(f"Unrecognized population member type {member_to_extract}.")
        self.model_template = None
        self.param_template = None
        self.param_shaper = None
        self.member_type =  member_to_extract

    def init(self, model, _):
        model_params, model_template  = model.partition()
        self.param_shaper = ex.core.ParameterReshaper(model_params, verbose=False)
        self.model_template = model_template
        self.param_template = model_params

    def __call__(self, training_state: PyTree):
        if self.param_shaper is None:
            raise RuntimeError
        if self.member_type == "mean":
            member = training_state[0].mean
        elif self.member_type == "best":
            member = training_state[0].best_member
        else:
            raise ValueError

        best_params = self.param_shaper.reshape_single(member)
        return eqx.combine(best_params, self.model_template)
