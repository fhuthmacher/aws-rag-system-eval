from __future__ import annotations

import typing as t

from langchain.chat_models import BedrockChat
from langchain.chat_models.base import BaseChatModel
from langchain.llms import Bedrock
from langchain.llms.base import BaseLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema import LLMResult

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks


def generate(
    prompts: list[ChatPromptTemplate],
    llm: BaseLLM | BaseChatModel,  # Bedrock only
    n: int = 1,
    temperature: int = 0,
    callbacks: t.Optional[Callbacks] = None,
) -> LLMResult:
    if isinstance(llm, BaseLLM):
        ps = [p.format() for p in prompts]
        result = llm.generate(ps, callbacks=callbacks)
    elif isinstance(llm, BaseChatModel):
        ps = [p.format_messages() for p in prompts]
        result = llm.generate(ps, callbacks=callbacks)

    return result
