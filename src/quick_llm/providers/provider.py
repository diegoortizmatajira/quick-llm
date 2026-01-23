"""Base classes and configurations for language model providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.language_models import (
    LanguageModelLike,
)


@dataclass
class ProviderSettings:
    """
    Base configuration settings for language model providers.
    """

    max_tokens: int = 4000
    """
    The maximum number of tokens that the language model can process in a single call.
    This serves as an upper limit for the length of the input and output combined.
    """
    temperature: float = 0.1
    """
    The sampling temperature for the language model, controlling the randomness of its output.
    Lower values make the output more deterministic, while higher values increase randomness.
    """


class Provider[SettingsVar: ProviderSettings](ABC):
    """
    Abstract base class for implementing various language model providers.

    Each provider should use a specific `SettingsType` extending `ProviderSettings`
    to configure and initialize its language model.

    Attributes:
        registry (dict[str, "Provider"]): A registry to store provider instances by name.
    """

    __registry: dict[str, "Provider"] = {}
    """
    A class-level registry to store provider instances by name.

    This dictionary maps unique provider names (str) to their respective Provider instances.
    It is used for managing and retrieving providers across the application.
    """

    def __init__(self, name: str):
        self._llm: LanguageModelLike | None = None
        self.register(name, self)

    @abstractmethod
    def build(self, settings: SettingsVar):
        """
        Build and configure the language model based on the provided settings.
        It is expected that each provider implementation will define how to build its specific language model.

        Args:
            settings (SettingsType): The settings required to configure and initialize the provider.
        """
        raise NotImplementedError()

    @property
    def llm(self) -> LanguageModelLike:
        """
        Retrieve the built language model instance.

        Returns:
            LanguageModelLike: The language model associated with the provider.

        Raises:
            ValueError: If the language model has not been built using the build() method.
        """
        if self._llm is None:
            raise ValueError("Provider not built. Call build() first.")
        return self._llm

    @classmethod
    def register(cls, name: str, provider: "Provider"):
        """
        Register a provider with a specific name.

        Args:
            name (str): The unique name to associate with the provider.
            provider (Provider): The provider instance to register.
        """
        cls.__registry[name] = provider

    @classmethod
    def get_by_name(cls, name: str) -> "Provider":
        """
        Retrieve a registered provider by its name.

        Args:
            name (str): The name of the provider to retrieve.

        Returns:
            Provider: The provider instance associated with the given name.

        Raises:
            ValueError: If the provider with the specified name is not found.
        """
        if name not in cls.__registry:
            raise ValueError(f"Provider '{name}' not found.")
        return cls.__registry[name]
