from typing import Self, override

from langchain_core.runnables.base import Runnable

from quick_llm.support import BaseFactory, ChainInputType, ChainOutputVar
from .base_assembler import BaseAssembler


class DefaultAssembler(BaseAssembler[ChainOutputVar]):
    """
    A default assembler that constructs a simple flow by chaining the prompt input adapter,
    the adapted language model, and the output transformer in sequence.
    """

    @override
    def assemble(self) -> Runnable[ChainInputType, ChainOutputVar]:
        return (
            self.wrap(self.factory.input_transformer, "Input Transformer")
            | self.wrap(
                self.factory.strategy.prompt_input_adapter,
                "Strategy Prompt Input Adapter",
            )
            | self.wrap(self.factory.prompt_template, "Prompt Template")
            | self.wrap(self.factory.strategy.adapted_llm, "Strategy Adapted LLM")
            # | self.wrap(self.language_model, "Language Model")
            # | self.wrap(self.output_cleaner, "Output Cleaner")
            # | self.wrap(self.output_transformer, "Output Transformer")
        )

    @override
    @classmethod
    def should_be_selected(cls, _factory: BaseFactory) -> Self | None:
        """
        The default assembler is selected if no other assembler is suitable for
        the given factory context.
        """
        return cls(_factory)
