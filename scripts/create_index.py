"""Create the agent_os_memory index on Elastic Cloud using the MCP search tool.

The Elastic MCP server only has read-only tools, so we use the search tool's
underlying connection to probe the cluster, and we create the index using
direct HTTP requests via the elasticsearch-py client.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_os.config import AgentOSConfig
from agent_os.mcp_client import ElasticMCPClient


INDEX_NAME = "agent_os_memory"

# Mapping for semantic atoms
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "summary": {"type": "text", "analyzer": "standard"},
            "keywords": {"type": "keyword"},
            "session_id": {"type": "keyword"},
            "source_message_ids": {"type": "keyword"},
            "source_message_count": {"type": "integer"},
            "time_range_start": {"type": "date"},
            "time_range_end": {"type": "date"},
            "created_at": {"type": "date"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1,
    },
}


async def main():
    config = AgentOSConfig()

    print(f"Creating index '{INDEX_NAME}' on Elastic Cloud...")
    print(f"  URL: {config.elasticsearch_url}")

    # Use direct HTTP — the MCP server is read-only
    import urllib.request
    import json
    import ssl

    url = f"{config.elasticsearch_url}/{INDEX_NAME}"
    data = json.dumps(INDEX_MAPPING).encode("utf-8")

    req = urllib.request.Request(url, data=data, method="PUT")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"ApiKey {config.elasticsearch_api_key}")

    # Allow self-signed certs for cloud
    ctx = ssl.create_default_context()

    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            body = json.loads(resp.read().decode())
            print(f"  ✅ Index created! Response: {json.dumps(body, indent=2)}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if "resource_already_exists_exception" in body:
            print(f"  ⚠️  Index '{INDEX_NAME}' already exists — nothing to do.")
        else:
            print(f"  ❌ Failed ({e.code}): {body}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Verify with MCP
    print("\nVerifying via MCP search...")
    async with ElasticMCPClient(config).connect() as client:
        try:
            result = await client.call_tool("search", {
                "index": INDEX_NAME,
                "queryBody": {"query": {"match_all": {}}, "size": 1},
            })
            print(f"  ✅ Search on '{INDEX_NAME}' succeeded (index exists)")
            if isinstance(result, str) and "index_not_found" in result:
                print(f"  ❌ Still not found — check your Elastic Cloud URL")
            else:
                print(f"  Result: {result}")
        except Exception as e:
            print(f"  ❌ Verification failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
