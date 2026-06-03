"""Diagnostic script — verifies each component of the Agentic Memory OS pipeline."""

import asyncio
import json
import sys
import os

# Ensure we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_os.config import AgentOSConfig
from agent_os.mcp_client import ElasticMCPClient


async def main():
    config = AgentOSConfig()

    print("=" * 60)
    print("  🔍 Agentic Memory OS — Diagnostics")
    print("=" * 60)

    # 1. Check config loaded correctly
    print("\n[1/5] Config loaded from .env")
    print(f"  ELASTICSEARCH_URL = {config.elasticsearch_url[:50]}...")
    print(f"  ES_URL (mapped)   = {config.elastic_env.get('ES_URL', 'MISSING')[:50]}...")
    print(f"  ES_API_KEY set?   = {bool(config.elastic_env.get('ES_API_KEY'))}")
    print(f"  Kernel model      = {config.kernel_model}")
    print(f"  Router model      = {config.router_model}")
    print(f"  Elastic index     = {config.elastic_index}")
    print("  ✅ Config OK")

    # 2. Connect to MCP
    print("\n[2/5] Connecting to Elastic MCP server...")
    async with ElasticMCPClient(config).connect() as client:
        print(f"  ✅ MCP connected!")
        tools = client.available_tools
        tool_names = [t["name"] for t in tools]
        print(f"  Available tools: {tool_names}")

        # 3. List indices
        print("\n[3/5] Listing existing Elasticsearch indices...")
        try:
            indices_result = await client.call_tool("list_indices", {})
            print(f"  ✅ list_indices succeeded")
            if isinstance(indices_result, str):
                print(f"  Indices:\n{indices_result[:500]}")
            else:
                print(f"  Indices: {json.dumps(indices_result, indent=2)[:500]}")
        except Exception as e:
            print(f"  ❌ list_indices failed: {e}")

        # 4. Try searching the target index
        print(f"\n[4/5] Searching index '{config.elastic_index}'...")
        try:
            search_result = await client.call_tool("search", {
                "index": config.elastic_index,
                "queryBody": {"query": {"match_all": {}}, "size": 1},
            })
            print(f"  ✅ Search succeeded")
            if isinstance(search_result, str):
                print(f"  Result:\n{search_result[:500]}")
            else:
                print(f"  Result: {json.dumps(search_result, indent=2)[:500]}")
        except Exception as e:
            error_str = str(e)
            if "index_not_found" in error_str:
                print(f"  ⚠️  Index '{config.elastic_index}' does NOT exist yet.")
                print(f"     This is expected on first run — it will be auto-created.")
            else:
                print(f"  ❌ Search failed: {e}")

        # 5. Check if 'index' tool exists (for writing)
        print(f"\n[5/5] Checking write capability...")
        if "index" in tool_names:
            print(f"  ✅ 'index' tool available — write operations supported")
        else:
            print(f"  ⚠️  No 'index' tool — MCP server is read-only")
            print(f"     Available tools: {tool_names}")
            print(f"     → Daemon will use in-memory atom storage as fallback")

    print("\n" + "=" * 60)
    print("  Diagnostic complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
