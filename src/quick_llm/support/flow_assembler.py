"""Flow Assembler Interface"""

from abc import ABC, abstractmethod
from typing import Generic, Self

from langchain_core.runnables import Runnable


from .type_definitions import ChainOutputVar, ChainInputType

from .base_factory import BaseFactory


class FlowAssembler(ABC, Generic[ChainOutputVar]):
    """
    A base abstract flow assembler interface for constructing complex flows from modular components.
    """

    @abstractmethod
    def assemble(self) -> Runnable[ChainInputType, ChainOutputVar]:
        """
        Assembles and returns a runnable that represents the complete flow.
        """

    @classmethod
    def should_be_selected(cls, _factory: BaseFactory) -> Self | None:
        """
        Determines whether this flow assembler should be selected based on the current context.
        """
        return None
