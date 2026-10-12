from dataclasses import dataclass

from lisjong.policies.minimal import MinimalPolicy

from ._parameters import DEFAULT_PARAMETER


@dataclass(frozen=True)
class CleanPackagePolicy(MinimalPolicy):
    parameter: int = DEFAULT_PARAMETER
