from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from voicebot.agent import build_agent
from voicebot.audio_io import play_audio, record_audio_until_silence


def build_voicebot():
    builder = StateGraph(MessagesState)
    builder.add_node("audio_input", record_audio_until_silence)
    builder.add_node("agent", build_agent())
    builder.add_node("audio_output", play_audio)
    builder.add_edge(START, "audio_input")
    builder.add_edge("audio_input", "agent")
    builder.add_edge("agent", "audio_output")
    builder.add_edge("audio_output", "audio_input")
    return builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    graph = build_voicebot()
    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 1000}
    for chunk in graph.stream(
        {"messages": [HumanMessage(content="Follow the user's instructions:")]},
        config=config,
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()
