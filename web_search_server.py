# web_search_server.py
# Provided to you. Imagine another team built and published this server.
# You will connect to it, discover its tool, and call it, without writing the tool yourself.
import os
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

mcp = FastMCP("Web Search")

@mcp.tool()
def web_search(query: str) -> str:
    """
    Search the web for current information on any topic.
    Use this for questions about recent events, current news, or anything
    unlikely to be in a static internal document corpus.
    Returns web page excerpts with their source URLs.
    """
    client = TavilyClient()  # reads TAVILY_API_KEY from the environment
    response = client.search(query=query, max_results=3, search_depth="basic")
    results = response.get("results", [])
    if not results:
        return "No web results found."
    return "\n\n".join(
        f"[Source: {r.get('url', 'unknown')}]\n{r.get('content', '')}"
        for r in results
    )


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
