from typing import Annotated

import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from .config import TIMEZONE


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def get_date_and_time() -> dict:
    """Fetch the current date and time from an API."""
    try:
        response = requests.get(
            f"https://timeapi.io/api/Time/current/zone?timeZone={TIMEZONE}",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return {"date_time": data["dateTime"], "timezone": data["timeZone"]}
    except requests.RequestException as e:
        return {"error": str(e)}


tools = [get_date_and_time]
llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def build_agent():
    """Chat agent: chatbot node that can loop through a tool node."""
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "chatbot")
    # Checkpointer lives on the parent graph only.
    return builder.compile()
