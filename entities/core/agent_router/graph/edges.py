from langgraph.graph import END
from typing_extensions import Literal

from entities.core.agent_router.graph import AgentMessagesState
from external.dependencies.settings import settings


def should_summarize_conversation(
    state: AgentMessagesState,
) -> Literal["summarize_conversation_node", "__end__"]:
    messages = state["messages"]

    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"

    return END


def select_workflow(
    state: AgentMessagesState,
) -> Literal["conversation_node", "image_node", "audio_node"]:
    workflow = state["workflow"]

    if workflow == "image":
        return "image_node"

    elif workflow == "audio":
        return "audio_node"

    else:
        return "conversation_node"
