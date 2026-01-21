"""Tests for the PromptInputParser class"""
import asyncio

import pytest
from pydantic import BaseModel, Field

from quick_llm.prompt_input_parser import PromptInputParser

# Test data constants
TEST_PARAM_NAME = "input"
TEST_CUSTOM_PARAM_NAME = "query"
TEST_STRING_VALUE = "Test input string"
TEST_DICT_VALUE = {"input": "Test input", "additional": "data"}


class TestModel(BaseModel):
    """Test Pydantic model for input parsing"""

    question: str = Field(description="The question")
    context: str = Field(description="Additional context")


class TestTransformValue:
    """Test transform_value method"""

    def test_transform_value_with_string(self):
        """Test transforming a string input"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        result = parser.transform_value(TEST_STRING_VALUE)

        assert isinstance(result, dict)
        assert result[TEST_PARAM_NAME] == TEST_STRING_VALUE
        assert len(result) == 1

    def test_transform_value_with_dict(self):
        """Test transforming a dictionary input (passthrough)"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        result = parser.transform_value(TEST_DICT_VALUE)

        assert result == TEST_DICT_VALUE
        assert result is TEST_DICT_VALUE  # Should be the same object

    def test_transform_value_with_basemodel(self):
        """Test transforming a Pydantic BaseModel input"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        model = TestModel(question="What is AI?", context="Technology discussion")

        result = parser.transform_value(model)

        assert isinstance(result, dict)
        assert result["question"] == "What is AI?"
        assert result["context"] == "Technology discussion"

    def test_transform_value_with_integer(self):
        """Test transforming an integer input"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        # Test with an integer input even though it's not a typical use case
        result = parser.transform_value(42)  # pyright: ignore[reportArgumentType]

        assert isinstance(result, dict)
        assert result[TEST_PARAM_NAME] == 42

    def test_transform_value_with_list(self):
        """Test transforming a list input"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        test_list = ["item1", "item2", "item3"]

        result = parser.transform_value(test_list)

        assert isinstance(result, dict)
        assert result[TEST_PARAM_NAME] == test_list

    def test_transform_value_with_none(self):
        """Test transforming None input"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        # Test with None input even though it's not a typical use case
        result = parser.transform_value(None)  # pyright: ignore[reportArgumentType]

        assert isinstance(result, dict)
        assert result[TEST_PARAM_NAME] is None

    def test_transform_value_with_custom_param_name(self):
        """Test transform with custom parameter name"""
        parser = PromptInputParser(TEST_CUSTOM_PARAM_NAME)

        result = parser.transform_value(TEST_STRING_VALUE)

        assert TEST_CUSTOM_PARAM_NAME in result
        assert result[TEST_CUSTOM_PARAM_NAME] == TEST_STRING_VALUE
        assert TEST_PARAM_NAME not in result

    def test_transform_value_with_empty_string(self):
        """Test transforming empty string"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        result = parser.transform_value("")

        assert isinstance(result, dict)
        assert result[TEST_PARAM_NAME] == ""

    def test_transform_value_with_empty_dict(self):
        """Test transforming empty dictionary"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        result = parser.transform_value({})

        assert result == {}

    def test_transform_value_with_nested_dict(self):
        """Test transforming nested dictionary"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        nested = {"outer": {"inner": "value"}, "list": [1, 2, 3]}

        result = parser.transform_value(nested)

        assert result == nested
        assert result["outer"]["inner"] == "value"


class TestInputParser:
    """Test input_parser method (synchronous iterator)"""

    def test_input_parser_single_string(self):
        """Test input_parser with single string value"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        input_iter = iter([TEST_STRING_VALUE])

        results = list(parser.input_parser(input_iter))

        assert len(results) == 1
        assert results[0][TEST_PARAM_NAME] == TEST_STRING_VALUE

    def test_input_parser_multiple_values(self):
        """Test input_parser with multiple values"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        values = ["first", "second", "third"]
        input_iter = iter(values)

        results = list(parser.input_parser(input_iter))

        assert len(results) == 3
        for i, value in enumerate(values):
            assert results[i][TEST_PARAM_NAME] == value

    def test_input_parser_mixed_types(self):
        """Test input_parser with mixed input types"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        values = ["string", {"input": "dict"}, TestModel(question="Q", context="C"), 42]
        input_iter = iter(values)

        results = list(parser.input_parser(input_iter))

        assert len(results) == 4
        assert results[0][TEST_PARAM_NAME] == "string"
        assert results[1]["input"] == "dict"
        assert results[2]["question"] == "Q"
        assert results[3][TEST_PARAM_NAME] == 42

    def test_input_parser_empty_iterator(self):
        """Test input_parser with empty iterator"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        input_iter = iter([])

        results = list(parser.input_parser(input_iter))

        assert len(results) == 0

    def test_input_parser_is_lazy(self):
        """Test that input_parser is lazy (generator-based)"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        def value_generator():
            yield "value1"
            yield "value2"
            # This should never be called if we only take first value
            raise RuntimeError("Should not reach here")

        gen = parser.input_parser(value_generator())
        first = next(gen)

        assert first[TEST_PARAM_NAME] == "value1"
        # If we don't consume the rest, the error shouldn't be raised


class TestAInputParser:
    """Test ainput_parser method (asynchronous iterator)"""

    @pytest.mark.asyncio
    async def test_ainput_parser_single_string(self):
        """Test ainput_parser with single string value"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        async def async_gen():
            yield TEST_STRING_VALUE

        results = []
        async for result in parser.ainput_parser(async_gen()):
            results.append(result)

        assert len(results) == 1
        assert results[0][TEST_PARAM_NAME] == TEST_STRING_VALUE

    @pytest.mark.asyncio
    async def test_ainput_parser_multiple_values(self):
        """Test ainput_parser with multiple values"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        values = ["first", "second", "third"]

        async def async_gen():
            for v in values:
                yield v

        results = []
        async for result in parser.ainput_parser(async_gen()):
            results.append(result)

        assert len(results) == 3
        for i, value in enumerate(values):
            assert results[i][TEST_PARAM_NAME] == value

    @pytest.mark.asyncio
    async def test_ainput_parser_mixed_types(self):
        """Test ainput_parser with mixed input types"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        async def async_gen():
            yield "string"
            yield {"input": "dict"}
            yield TestModel(question="Q", context="C")
            yield 42

        results = []
        # Test with mixed input types even though it's not a typical use case
        async for result in parser.ainput_parser(async_gen()):  # pyright: ignore[reportArgumentType]
            results.append(result)

        assert len(results) == 4
        assert results[0][TEST_PARAM_NAME] == "string"
        assert results[1]["input"] == "dict"
        assert results[2]["question"] == "Q"
        assert results[3][TEST_PARAM_NAME] == 42

    @pytest.mark.asyncio
    async def test_ainput_parser_empty_iterator(self):
        """Test ainput_parser with empty async iterator"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        async def async_gen():
            return
            yield  # pragma: no cover

        results = []
        async for result in parser.ainput_parser(async_gen()):
            results.append(result)

        assert len(results) == 0


class TestInvoke:
    """Test invoke method"""

    def test_invoke_with_string(self):
        """Test invoke with string input"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        result = parser.invoke(TEST_STRING_VALUE)

        assert isinstance(result, dict)
        assert result[TEST_PARAM_NAME] == TEST_STRING_VALUE

    def test_invoke_with_dict(self):
        """Test invoke with dictionary input"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        result = parser.invoke(TEST_DICT_VALUE)

        assert result == TEST_DICT_VALUE

    def test_invoke_with_basemodel(self):
        """Test invoke with Pydantic BaseModel"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        model = TestModel(question="What?", context="Why?")

        result = parser.invoke(model)

        assert isinstance(result, dict)
        assert result["question"] == "What?"
        assert result["context"] == "Why?"

    def test_invoke_merges_multiple_stream_outputs(self):
        """Test that invoke merges multiple stream outputs correctly"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        # The invoke method streams and merges outputs
        # For a single value, there should be only one output
        result = parser.invoke(TEST_STRING_VALUE)

        assert isinstance(result, dict)
        assert TEST_PARAM_NAME in result

    def test_invoke_with_custom_param_name(self):
        """Test invoke with custom parameter name"""
        parser = PromptInputParser(TEST_CUSTOM_PARAM_NAME)

        result = parser.invoke(TEST_STRING_VALUE)

        assert TEST_CUSTOM_PARAM_NAME in result
        assert result[TEST_CUSTOM_PARAM_NAME] == TEST_STRING_VALUE


class TestAInvoke:
    """Test ainvoke method (async)"""

    @pytest.mark.asyncio
    async def test_ainvoke_with_string(self):
        """Test ainvoke with string input"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        result = await parser.ainvoke(TEST_STRING_VALUE)

        assert isinstance(result, dict)
        assert result[TEST_PARAM_NAME] == TEST_STRING_VALUE

    @pytest.mark.asyncio
    async def test_ainvoke_with_dict(self):
        """Test ainvoke with dictionary input"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        result = await parser.ainvoke(TEST_DICT_VALUE)

        assert result == TEST_DICT_VALUE

    @pytest.mark.asyncio
    async def test_ainvoke_with_basemodel(self):
        """Test ainvoke with Pydantic BaseModel"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        model = TestModel(question="What?", context="Why?")

        result = await parser.ainvoke(model)

        assert isinstance(result, dict)
        assert result["question"] == "What?"
        assert result["context"] == "Why?"

    @pytest.mark.asyncio
    async def test_ainvoke_merges_multiple_stream_outputs(self):
        """Test that ainvoke merges multiple async stream outputs correctly"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        result = await parser.ainvoke(TEST_STRING_VALUE)

        assert isinstance(result, dict)
        assert TEST_PARAM_NAME in result


class TestStream:
    """Test stream method"""

    def test_stream_with_string(self):
        """Test stream with string input"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        results = list(parser.stream(TEST_STRING_VALUE))

        assert len(results) == 1
        assert results[0][TEST_PARAM_NAME] == TEST_STRING_VALUE

    def test_stream_with_dict(self):
        """Test stream with dictionary input"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        results = list(parser.stream(TEST_DICT_VALUE))

        assert len(results) == 1
        assert results[0] == TEST_DICT_VALUE

    def test_stream_with_basemodel(self):
        """Test stream with Pydantic BaseModel"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        model = TestModel(question="Q1", context="C1")

        results = list(parser.stream(model))

        assert len(results) == 1
        assert results[0]["question"] == "Q1"
        assert results[0]["context"] == "C1"

    def test_stream_returns_generator(self):
        """Test that stream returns a generator"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        stream = parser.stream(TEST_STRING_VALUE)

        # Should be a generator/iterator
        assert hasattr(stream, "__iter__")
        assert hasattr(stream, "__next__")


class TestAStream:
    """Test astream method (async)"""

    @pytest.mark.asyncio
    async def test_astream_with_string(self):
        """Test astream with string input"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        results = []
        async for result in parser.astream(TEST_STRING_VALUE):
            results.append(result)

        assert len(results) == 1
        assert results[0][TEST_PARAM_NAME] == TEST_STRING_VALUE

    @pytest.mark.asyncio
    async def test_astream_with_dict(self):
        """Test astream with dictionary input"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        results = []
        async for result in parser.astream(TEST_DICT_VALUE):
            results.append(result)

        assert len(results) == 1
        assert results[0] == TEST_DICT_VALUE

    @pytest.mark.asyncio
    async def test_astream_with_basemodel(self):
        """Test astream with Pydantic BaseModel"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        model = TestModel(question="Q1", context="C1")

        results = []
        async for result in parser.astream(model):
            results.append(result)

        assert len(results) == 1
        assert results[0]["question"] == "Q1"
        assert results[0]["context"] == "C1"


class TestEdgeCases:
    """Test edge cases and special scenarios"""

    def test_with_boolean_input(self):
        """Test with boolean input"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        # Test with a boolean input even though it's not a typical use case
        result = parser.invoke(True)  # pyright: ignore[reportArgumentType]

        assert result[TEST_PARAM_NAME] is True

        # Test with a boolean input even though it's not a typical use case
        result = parser.invoke(False)  # pyright: ignore[reportArgumentType]

        assert result[TEST_PARAM_NAME] is False

    def test_with_float_input(self):
        """Test with float input"""
        parser = PromptInputParser(TEST_PARAM_NAME)


        # Test with a float input even though it's not a typical use case
        result = parser.invoke(3.14159)  # pyright: ignore[reportArgumentType]

        assert result[TEST_PARAM_NAME] == 3.14159

    def test_with_complex_nested_structure(self):
        """Test with complex nested data structure"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        complex_data = {
            "level1": {
                "level2": {"level3": "deep value"},
                "list": [1, 2, {"nested": True}],
            }
        }

        result = parser.invoke(complex_data)

        assert result == complex_data
        assert result["level1"]["level2"]["level3"] == "deep value"

    def test_with_special_characters_in_string(self):
        """Test with special characters in string input"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        special = "Test with émojis: 🎉 and symbols: !@#$%^&*()_+-=[]{}|;':\",./<>?"

        result = parser.invoke(special)

        assert result[TEST_PARAM_NAME] == special

    def test_with_multiline_string(self):
        """Test with multiline string input"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        multiline = """Line 1
        Line 2
        Line 3"""

        result = parser.invoke(multiline)

        assert result[TEST_PARAM_NAME] == multiline

    def test_with_very_long_string(self):
        """Test with very long string input"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        long_string = "x" * 10000

        result = parser.invoke(long_string)

        assert result[TEST_PARAM_NAME] == long_string
        assert len(result[TEST_PARAM_NAME]) == 10000

    def test_dict_with_conflicting_param_name(self):
        """Test dictionary that already contains the param name"""
        parser = PromptInputParser(TEST_PARAM_NAME)
        dict_with_input = {"input": "original", "other": "data"}

        result = parser.invoke(dict_with_input)

        # Dict should pass through unchanged
        assert result == dict_with_input
        assert result["input"] == "original"

    def test_basemodel_with_optional_fields(self):
        """Test BaseModel with optional fields"""

        class OptionalModel(BaseModel):
            required: str
            optional: str | None = None

        parser = PromptInputParser(TEST_PARAM_NAME)
        model1 = OptionalModel(required="value")

        result1 = parser.invoke(model1)

        assert result1["required"] == "value"
        assert result1["optional"] is None

        model2 = OptionalModel(required="value", optional="present")
        result2 = parser.invoke(model2)

        assert result2["required"] == "value"
        assert result2["optional"] == "present"

    def test_basemodel_with_nested_models(self):
        """Test BaseModel with nested Pydantic models"""

        class InnerModel(BaseModel):
            inner_field: str

        class OuterModel(BaseModel):
            outer_field: str
            nested: InnerModel

        parser = PromptInputParser(TEST_PARAM_NAME)
        model = OuterModel(outer_field="outer", nested=InnerModel(inner_field="inner"))

        result = parser.invoke(model)

        assert result["outer_field"] == "outer"
        assert result["nested"]["inner_field"] == "inner"


class TestIntegration:
    """Integration tests combining multiple methods"""

    def test_sync_and_async_produce_same_results(self):
        """Test that sync and async methods produce identical results"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        sync_result = parser.invoke(TEST_STRING_VALUE)


        async_result = asyncio.run(parser.ainvoke(TEST_STRING_VALUE))

        assert sync_result == async_result

    def test_stream_and_invoke_consistency(self):
        """Test that stream and invoke produce consistent results"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        invoke_result = parser.invoke(TEST_STRING_VALUE)
        stream_results = list(parser.stream(TEST_STRING_VALUE))

        # For single input, stream should produce one output
        assert len(stream_results) == 1
        assert stream_results[0] == invoke_result

    @pytest.mark.asyncio
    async def test_astream_and_ainvoke_consistency(self):
        """Test that astream and ainvoke produce consistent results"""
        parser = PromptInputParser(TEST_PARAM_NAME)

        ainvoke_result = await parser.ainvoke(TEST_STRING_VALUE)

        astream_results = []
        async for result in parser.astream(TEST_STRING_VALUE):
            astream_results.append(result)

        assert len(astream_results) == 1
        assert astream_results[0] == ainvoke_result

    def test_different_param_names_produce_different_keys(self):
        """Test that different param names create different dictionary keys"""
        parser1 = PromptInputParser("param1")
        parser2 = PromptInputParser("param2")

        result1 = parser1.invoke(TEST_STRING_VALUE)
        result2 = parser2.invoke(TEST_STRING_VALUE)

        assert "param1" in result1
        assert "param2" in result2
        assert result1["param1"] == result2["param2"]
