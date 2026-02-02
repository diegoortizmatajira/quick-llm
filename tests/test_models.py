"""Test models for validating output structures."""

from typing import TypedDict
from pydantic import BaseModel, Field


class AnswerOutput(BaseModel):
    """Sample object structure to test the JSON parsing feature"""

    what: str = Field(description="Summarizes/rephrase the question being answered.")
    when: str = Field(
        description="Provides a date-formatted answer to the question when required."
    )
    who: str = Field(
        description="Provides a proper name answer to the question when required."
    )
    general: str = Field(description="Provides a short-text answer to the question.")


class TestOutput(BaseModel):
    answer: str


class TestOutputDictionary(TypedDict):
    answer: str
