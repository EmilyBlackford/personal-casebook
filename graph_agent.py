"""
The same retrieve/web_search agent, restructured as a LangGraph StateGraph.

Copy your completed pipeline.py from Week 9 Day 2 into this directory.

Usage:
    python graph_agent.py
"""
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langfuse import observe
from pipeline import load_vector_store, retrieve as pipeline_retrieve


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


vector_store = load_vector_store()


@tool
def retrieve(query: str) -> str:
    """
        Search a corpus of four AI governance documents: the EU AI Act, the
        International AI Safety Act, the NIST AI Risk Management Framework, 
        and the Meridian AI Governance report. Use this for any question about 
        AI Regulation, policy, standards, or governance.
        Returns a list of text passages with source document names.
    """
    # Call the pipeline function from pipeline.py
    passages = pipeline_retrieve(query, vector_store)

    # Handle the empty case
    if not passages:
        return "No relevant passages found."
    return "\n\n".join(
        # Format each passage (note 'score' was deliberately not included)
        f"[Source: {p['source']}]\n{p['content']}" for p in passages
    )


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information on any topic.
    Use this for questions about recent events, current news, or information
    that is unlikely to be in the internal document corpus.
    Returns web page excerpts with source URLs.
    """
    from tavily import TavilyClient
    client = TavilyClient()
    response = client.search(query=query, max_results=3, search_depth="basic")
    results = response.get("results", [])
    if not results:
        return "No web results found."
    parts = []
    for r in results:
        parts.append(f"[Source: {r.get('url', 'unknown')}]\n{r.get('content', '')}")
    return "\n\n".join(parts)


tools = [retrieve, web_search]
tool_node = ToolNode(tools)

SYSTEM_PROMPT = """You are a research assistant with access to an internal document corpus and web search.
Use the 'retrieve' tool for questions about the internal knowledge base documents.
Use the 'web_search' tool for current news or information outside the corpus.
For complex questions, break them into sub-questions and call the relevant tool for each part.
Always cite the source of each piece of information in your final answer."""

model = ChatVertexAI(model_name="gemini-2.5-flash").bind_tools(tools)

@observe()
def call_model(state: AgentState, model) -> dict:
    """Prepend SYSTEM_PROMPT if the state has no system message yet, then invoke `model`.

    `model` is passed in by build_graph (see below) rather than read from a
    module global, so Day 5 can rebuild the same graph with a different set of
    bound tools. Returns {"messages": [response]}.
    """
    # Grab the current message history from state["messages"]
    # This is the running conversation including human messages, AI messages, 
    # tool results, all accumulated by the add_messages reducer
    messages = state["messages"]

    # Check for an existing system message. Loop through with isinstance()
    # This matters because call_model runs in a loop (model, tools, model, tools)
    if not any(isinstance(m, SystemMessage) for m in messages):
        # If no SystemMessage is found, build a new list with SYSTEM_PROMPT at the front
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = model.invoke(messages)
    # Return a list is rewuired because add_messages expects an iterable to merge into the existing history
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Return "call_tools" if the last message has tool_calls, else END."""
    if state["messages"][-1].tool_calls:
        return "call_tools"
    else:
        return END


def build_graph(model, tool_node):
    """Wire call_model and call_tools (a ToolNode) into a StateGraph.

    Takes the bound `model` and the `tool_node` as parameters so the graph is
    not tied to module globals. Add call_model as a node that calls
    call_model(state, model); entry point "call_model"; conditional edge via
    should_continue; an edge from "call_tools" back to "call_model" for the
    cycle. Return the compiled graph.
    """
    graph = StateGraph(AgentState)
    graph.add_node("call_model", lambda state: call_model(state, model))
    graph.add_node("call_tools", tool_node)

    graph.set_entry_point("call_model")

    graph.add_conditional_edges(
        "call_model",
        should_continue,
        {"call_tools": "call_tools", END: END}
    )

    graph.add_edge("call_tools", "call_model")

    return graph.compile()


model = ChatVertexAI(model_name="gemini-2.5-flash").bind_tools(tools)
graph = build_graph(model, tool_node)


def extract_contexts_from_messages(messages: list) -> list[str]:
    """Return the .content of every ToolMessage in `messages`, in order.

    Used to bridge the message history into RAGAS's "contexts" field.
    """
    # Filter the `messages` list and keep only `ToolMessage` instances
    # Pull out `.content` for each `ToolMessage`
    return [m.content for m in messages if isinstance(m, ToolMessage)]


@observe()
def ask_agent(question: str) -> dict:
    """Run the graph on `question` and summarise the result.

    Returns {"question": str, "answer": str, "tool_calls": list[dict],
    "messages": list}, where each tool_calls entry is {"tool": ..., "args": ...}.
    """
    # Build the starting state, wrap the question in a `HumanMessage` and put it in 
    # the `messages` list
    initial_state = {"messages": [HumanMessage(content=question)]}
    # Run the graph
    final_state = graph.invoke(initial_state)

    #The last message in `final_state["messages"]` is the model's final response
    # (once it's done calling tools and has something to say)
    messages = final_state["messages"]
    answer = messages[-1].content

    # Loop through every message in the history and collect the non-empty `tool_calls`
    tool_calls = []
    for m in messages:
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                tool_calls.append({"tool": tc["name"], "args": tc["args"]})

    # Return the full shape in one dict
    return {
        "question": question,
        "answer": answer,
        "tool_calls": tool_calls,
        "messages": messages
    }


if __name__ == "__main__":
    result = ask_agent("How do bats locate prey, and what frequencies do they typically use?")
    print(result["answer"])
    print("\nTool calls:")
    for tc in result["tool_calls"]:
        print(f"  {tc['tool']}: {tc['args']}")

    print("\n=== Graph structure ===")
    print(graph.get_graph().draw_mermaid())
