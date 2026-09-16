import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools


async def explore_mcp_server():
    server_params = StdioServerParameters(
        command="python",
        args=["web_search_server.py"],
        env={"TAVILY_API_KEY": os.environ["TAVILY_API_KEY"]},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover: the tool and its description come from the server's
            # manifest, not from anything you wrote.
            tools = await load_mcp_tools(session)
            print("Discovered tools:")
            for t in tools:
                print(f"  {t.name}: {t.description}")

            # Call the tool once, through the protocol.
            result = await session.call_tool(
                "web_search", {"query": "AI protocol updates 2026"}
            )
            print("\nLive result through MCP:")
            print(result.content[0].text[:400])


if __name__ == "__main__":
    asyncio.run(explore_mcp_server())
