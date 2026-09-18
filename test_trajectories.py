"""
Pytest assertions on the agent's tool-routing behaviour.

Requires your completed graph_agent.py (Day 3) in the same directory.

Usage:
    pytest test_trajectories.py -v
"""
from graph_agent import ask_agent


def test_corpus_question_calls_retrieve():
    """A question answerable from the sample corpus should call `retrieve`."""
    # Property: a question about corpus content should call `retrieve`
    # Why: it's the only tool with access to the internal documents, and the 
    # system prompt directs the model to use it for such questions.
    result = ask_agent(
        "How do bats use echolocation to locate prey, and what frequency ranges are typically involved?"
    )

    tool_names = [tc["tool"] for tc in result["tool_calls"]]

    assert "retrieve" in tool_names, (
        f"Expected 'retrieve' to be called for a corpus question, "
        f"but tools called were: {tool_names}"
    )
    assert result["answer"], "Expected a non-empty final answer."


def test_current_events_calls_web_search():
    """A question about very recent events should call `web_search`."""
    # Property: a question about "today's news" should call `web_search`
    # Why: the static corpus cannot contain current events, 
    # so `web_search` is the only tool that can answer

    result = ask_agent(
        "What happened in the news today? Give me a current headline "
        "from the last 24 hours."
    )

    tool_names = [tc["tool"] for tc in result["tool_calls"]]

    assert "web_search" in tool_names, (
        f"Expected 'web_search' to be called for a current-events question, "
        f"but tools called were: {tool_names}"
    )
    assert result["answer"], "Expected a non-empty final answer."


def test_multi_part_corpus_question_retrieves_multiple_times():
    """A question spanning two corpus topics should retrieve at least once,
    and the answer should mention both topics."""
    # Property: a question spanning two corpus topics should call `retrieve` at least 
    # once, and the answer should cover both topics
    # Why: the system prompt says to split complex questions into sub-questions and 
    # call the relevant tool for each part, rather than dropping one

    result = ask_agent(
        "Compare how bats use echolocation with how bees communicate "
        "the location of food sources to their hive."
    )

    tool_names = [tc["tool"] for tc in result["tool_calls"]]
    retrieve_calls = [t for t in tool_names if t == "retrieve"]

    assert len(retrieve_calls) >= 1, (
        f"Expected at least one 'retrieve' call for a multi-part corpus "
        f"question, but tools called were: {tool_names}"
    )

    answer_lower = result["answer"].lower()
    assert "bat" in answer_lower, "Expected the answer to mention bats."
    assert "bee" in answer_lower, "Expected the answer to mention bees."
