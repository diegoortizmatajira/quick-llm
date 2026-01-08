from langchain_core.language_models import FakeListChatModel, FakeListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate

from quick_llm.chain_factory import ChainFactory

TEST_INPUT = "Test input"
TEST_EXPECTED_RESPONSE = "This is a sample response."


def test_chain_factory():
    models = [
        FakeListLLM(responses=[TEST_EXPECTED_RESPONSE]),
        FakeListChatModel(responses=[TEST_EXPECTED_RESPONSE]),
    ]
    for model in models:
        factory = (
            ChainFactory()
            .use_prompt_template(PromptTemplate.from_template("Sample Prompt {input}"))
            .use_language_model(model)
            .use_output_transformer(StrOutputParser())
        )
        chain = factory.build_raw_chain()
        response = chain.invoke(TEST_INPUT)
        assert response == TEST_EXPECTED_RESPONSE
