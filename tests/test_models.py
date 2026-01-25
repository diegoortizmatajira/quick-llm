"""Test models for validating output structures."""

from typing import TypedDict
from pydantic import BaseModel


class TestOutput(BaseModel):
    answer: str


class TestOutputDictionary(TypedDict):
    answer: str
