"""Module to parse prompt inputs into a standardized dictionary format."""

from typing import Any, AsyncIterator, Iterator, Optional, cast, override
from langchain_core.runnables import RunnableConfig, RunnableGenerator
from pydantic import BaseModel

from quick_llm.type_definitions import ChainInputType


class PromptInputParser(RunnableGenerator[ChainInputType, dict]):
    def __init__(self, prompt_input_param: str):
        super().__init__(self.input_parser, self.ainput_parser)
        self.prompt_input_param = prompt_input_param

    def transform_value(self, value: ChainInputType) -> dict:
        """
        Transforms the input value into a dictionary format.
        """
        input_dict = {}
        if isinstance(value, BaseModel):
            input_dict = value.model_dump()
        elif isinstance(value, dict):
            input_dict = value
        else:
            input_dict = {self.prompt_input_param: value}
        return input_dict

    def input_parser(self, input_value: Iterator[ChainInputType]) -> Iterator[dict]:
        """
        Parses any non dictionary value into a dictionary
        """
        for value in input_value:
            yield self.transform_value(value)

    async def ainput_parser(
        self, input_value: AsyncIterator[ChainInputType]
    ) -> AsyncIterator[dict]:
        """
        Parses any non dictionary value into a dictionary
        """
        async for value in input_value:
            yield self.transform_value(value)

    @override
    def invoke(
        self,
        input: ChainInputType,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> dict:
        final = None
        for output in self.stream(input, config, **kwargs):
            if final is None:
                final = output
            else:
                final = final | output
        return cast(dict, final)

    @override
    async def ainvoke(
        self,
        input: ChainInputType,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> dict:
        final = None
        async for output in self.astream(input, config, **kwargs):
            if final is None:
                final = output
            else:
                final = final | output
        return cast(dict, final)
