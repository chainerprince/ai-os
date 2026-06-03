"""Quick test — write a test document to agent_os_memory and read it back via MCP."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from elasticsearch import AsyncElasticsearch
from agent_os.config import AgentOSConfig
from agent_os.mcp_client import ElasticMCPClient


async def main():
    config = AgentOSConfig()

    # 1. Write directly via ES client
    es = AsyncElasticsearch(
        config.elasticsearch_url,
        api_key=config.elasticsearch_api_key,
    )

    test_doc = {
        "summary": "User discussed building an Agentic Memory OS for a hackathon using Gemini and Elasticsearch.",
        "keywords": ["hackathon", "memory", "gemini", "elasticsearch", "mcp"],
        "session_id": "test-session-001",
        "source_message_count": 5,
        "created_at": "2026-06-03T06:00:00Z",
    }

    print("📝 Writing test document to Elasticsearch...")
    result = await es.index(
        index=config.elastic_index,
        id="test-atom-001",
        document=test_doc,
    )
    print(f"   ✅ Write result: {result['result']}")

    # Force a refresh so the document is immediately searchable
    await es.indices.refresh(index=config.elastic_index)
    print("   🔄 Index refreshed")

    await es.close()

    # 2. Read it back via MCP search
    print("\n🔍 Reading back via MCP search...")
    async with ElasticMCPClient(config).connect() as client:
        result = await client.search(query="hackathon memory gemini")
        print(f"   ✅ Search returned {len(result)} result(s)")
        for i, doc in enumerate(result):
            print(f"   [{i}] {doc}")

    # 3. Cleanup
    print("\n🧹 Cleaning up test document...")
    es = AsyncElasticsearch(
        config.elasticsearch_url,
        api_key=config.elasticsearch_api_key,
    )
    await es.delete(index=config.elastic_index, id="test-atom-001", ignore=[404])
    await es.close()
    print("   ✅ Cleanup done")

    print("\n✅ Full write→read loop verified!")


if __name__ == "__main__":
    asyncio.run(main())
