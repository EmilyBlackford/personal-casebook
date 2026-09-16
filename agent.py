"""
A ReAct agent built directly on the google-genai SDK -- no LangGraph, no agent
executor library.

Copy your completed pipeline.py from Week 9 Day 2 into this directory before
running this file. Do not modify pipeline.py.

Usage:
    python agent.py
"""
import os
from google import genai
from google.genai import types
from pipeline import load_vector_store, retrieve
from web_search_server import web_search

client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location="europe-west4",
)

# Define retrieve_declaration, a types.FunctionDeclaration describing the
# retrieve() function above (name="retrieve" -- must match exactly, since
# execute_tool() below dispatches on this string -- a clear description, and
# a "query" string parameter), then wrap it in:
#   retrieval_tool = types.Tool(function_declarations=[retrieve_declaration])
retrieve_declaration = types.FunctionDeclaration(
    name="retrieve",
    description=(
        "Search a corpus of four AI governance documents: the EU AI Act, the"
        "International AI Safety Act, the NIST AI Risk Management Framework, "
        "and the Meridian AI Governance report. Use this for any question about "
        "AI Regulation, policy, standards, or governance."
        "Returns a list of text passages with source document names."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to run against the document corpus."
            }
        },
        "required": ["query"]
    }
)

retrieval_tool = types.Tool(function_declarations=[retrieve_declaration])

#The web search tool
web_search_declaration = types.FunctionDeclaration(
    name="web_search",
    description=(
        "Search the web for current or general information not found in the "
        "internal document corpus. Use this for recent news, events after the "
        "corpus's coverage, or any topic outside governance policy."
        "Returns a list of web page excerpts with source URLs."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to run against the web."
            }
        },
        "required": ["query"]
    }
)

both_tools = types.Tool(
    function_declarations=[retrieve_declaration, web_search_declaration]
    )

# The model can only request a call, this code executes it
def execute_tool(tool_name: str, tool_args: dict, vector_store) -> dict:
    """Run the tool the model requested and return its result.

    If tool_name == "retrieve", call retrieve() and return
    {"passages": ..., "count": ...}. Raise ValueError for anything else.
    """
    if tool_name == "retrieve":
        passages = retrieve(tool_args["query"], vector_store)
        return {
            "passages": passages, 
            "count": len(passages)
            }
    if tool_name == "web_search":
        results = web_search(tool_args["query"])
        return {
            "results": results, 
            "count": len(results)
            }
    raise ValueError(f"Unknown tool: {tool_name}")

def run_agent(query: str, vector_store, max_iterations: int = 5) -> dict:
    # Step 1: contents is the running transcript we'll keep resending. 
    # At the start, it's just the user's question.
    # Pass this into the response below
    contents = [types.Content(role="user", parts=[types.Part(text=query)])]
    tool_calls_log = []

    # Step 8: Give up gracefully if it never finishes
    for i in range(max_iterations):
        # Step 2: Call the model, passing the whole transcript and tell it which tools it's allowed to use
        # The model looks at contents and 
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(tools=[both_tools])
            )

        # Step 3: This is the loop's exit condition. If response.function_calls is empty, 
        # stop here and return.
        if not response.function_calls:
            return {
                "answer" : response.text,
                "tool_calls": tool_calls_log,
                "iterations": i + 1
            }

        # Step 4: If not done, record what the model just said.
        # If the model did ask for tool calls, you save its turn into the 
        # transcript, exactly once.
        contents.append(response.candidates[0].content)

        # Step 5: Loop over the requested calls and execute them, using the execute_tool above.
        reply_parts = []
        for call in response.function_calls:
            args = dict(call.args)
            result = execute_tool(call.name, args, vector_store)
            tool_calls_log.append({"tool": call.name, "args": args})
            reply_parts.append(types.Part.from_function_response(name=call.name, response=result))

        # Step 6: Send all the results back, packaged into one Content object
        contents.append(types.Content(role="user", parts=reply_parts))
        # Step 7: loops back to asking the model again. 
        # It now sees the original question, its own request, and your answer.
        # so it can decide whether it has enough or needs to call again.

    # Step 8: the graceful fallback message instead of crashing
    return {
        "answer": "Could not produce a final answer within the iteration limit.",
        "tool_calls": tool_calls_log,
        "iterations": max_iterations
    }

if __name__ == "__main__":
    vs = load_vector_store()

    # Step 1: a simple single-step query. Expect one call to `retrieve`,
    # then a final answer grounded in the retrieved passages.
    result = run_agent("Is there a plan for future AI interventions?", vs)
    print(result["answer"])
    print(f"\nTool calls made: {result['tool_calls']}")
    # print(f"Iterations: {result['iterations']}")

    # Step 2: a multi-part query. Does the model break it into sub-queries
    # and call `retrieve` more than once?
    # result = run_agent(
    #     "When was the EU AI Act published "
    #     "and how many types of risks are identified?",
    #     vs,
    # )
    # print(result["answer"])
    # print(f"\nTool calls: {result['tool_calls']}")
    # print(f"Iterations: {result['iterations']}")


