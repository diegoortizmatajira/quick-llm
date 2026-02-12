from abc import ABC
from typing import Any, Generic

from langchain_core.runnables.base import Runnable
from ..support import BaseFactory, ChainOutputVar, FlowAssembler


class BaseAssembler(Generic[ChainOutputVar], FlowAssembler[ChainOutputVar], ABC):
    """
    A base abstract assembler class for constructing complex flows from modular components.
    """

    def __init__(self, factory: BaseFactory[ChainOutputVar, Any]):
        self._factory = factory

    @property
    def factory(self) -> BaseFactory[ChainOutputVar, Any]:
        """Returns the chain factory associated with the strategy."""
        return self._factory

    def wrap[Input, Output](
        self, runnable: Runnable[Input, Output], caption: str
    ) -> Runnable[Input, Output]:
        """
        Wraps a runnable with a caption for better traceability.
        """
        return self.factory.wrap(runnable, caption)
