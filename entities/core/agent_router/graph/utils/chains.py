from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from entities.core.agent_router.graph.utils.helper import AsteriskRemovalParser, get_chat_model

from entities.core.prompt.prompts import CHARACTER_CARD_PROMPT, ROUTER_PROMPT,RouterResponseCls


def get_router_chain():
    model = get_chat_model(temperature=0.3).with_structured_output(RouterResponseCls)

    prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_PROMPT), MessagesPlaceholder(variable_name="messages")]
    )

    return prompt | model


def get_character_response_chain(summary: str = ""):
    model = get_chat_model()
    system_message = CHARACTER_CARD_PROMPT

    if summary:
        system_message += f"\n\nSummary of conversation earlier between Ava and the user: {summary}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | model | AsteriskRemovalParser()
