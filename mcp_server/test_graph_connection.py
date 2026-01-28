#!/usr/bin/env python3
"""
Test script to verify Graphiti MCP server is working with Gemini embeddings.
Creates test nodes and verifies they can be searched.

Usage:
    uv run python test_graph_connection.py [--url URL]

Examples:
    uv run python test_graph_connection.py
    uv run python test_graph_connection.py --url http://your-server:8001/mcp
"""

import asyncio
import json
import os
import sys
import time
from typing import Any

import httpx
import typer

app = typer.Typer()

# Default URL - can be overridden via --url or MCP_SERVER_URL env var
DEFAULT_URL = os.environ.get('MCP_SERVER_URL', 'http://localhost:8001/mcp')


async def call_mcp_tool(
    server_url: str, tool_name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Call an MCP tool via HTTP."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        # MCP tools are called via JSON-RPC style requests
        request_body = {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'tools/call',
            'params': {'name': tool_name, 'arguments': arguments},
        }

        response = await client.post(
            f'{server_url}/mcp', json=request_body, headers={'Content-Type': 'application/json'}
        )

        if response.status_code != 200:
            raise Exception(f'HTTP {response.status_code}: {response.text}')

        result = response.json()
        if 'error' in result:
            raise Exception(f'MCP Error: {result["error"]}')

        return result.get('result', result)


async def add_test_memories():
    """Add test memories to create nodes and edges."""

    test_group_id = 'graphiti_test_group'

    # Test memory 1: Alice works at TechCorp
    print('Adding memory 1: Alice works at TechCorp...')
    result1 = await call_mcp_tool(
        'add_memory',
        {
            'name': 'Employee Info 1',
            'episode_body': 'Alice Johnson is a senior software engineer at TechCorp. She has been working there for 5 years and specializes in Python development.',
            'group_id': test_group_id,
            'source': 'text',
            'source_description': 'test data',
        },
    )
    print(f'  Result: {result1}')

    # Test memory 2: Bob works at TechCorp with Alice
    print('\nAdding memory 2: Bob works at TechCorp...')
    result2 = await call_mcp_tool(
        'add_memory',
        {
            'name': 'Employee Info 2',
            'episode_body': 'Bob Smith is a project manager at TechCorp. He manages the team where Alice Johnson works. They collaborate on the CloudSync project.',
            'group_id': test_group_id,
            'source': 'text',
            'source_description': 'test data',
        },
    )
    print(f'  Result: {result2}')

    # Test memory 3: TechCorp builds CloudSync product
    print('\nAdding memory 3: TechCorp and CloudSync...')
    result3 = await call_mcp_tool(
        'add_memory',
        {
            'name': 'Company Info',
            'episode_body': 'TechCorp is a technology company that develops the CloudSync product. CloudSync is a cloud synchronization tool used by thousands of enterprises.',
            'group_id': test_group_id,
            'source': 'text',
            'source_description': 'test data',
        },
    )
    print(f'  Result: {result3}')

    return test_group_id


async def wait_for_processing(seconds: int = 15):
    """Wait for async processing to complete."""
    print(f'\nWaiting {seconds} seconds for async processing...')
    for i in range(seconds):
        print(f'  {seconds - i} seconds remaining...', end='\r')
        await asyncio.sleep(1)
    print(f'\nDone waiting.')


async def search_and_verify(group_id: str):
    """Search for the test nodes and verify they exist."""

    print('\n' + '=' * 60)
    print('SEARCHING FOR NODES')
    print('=' * 60)

    # Search for Alice
    print("\nSearching for 'Alice'...")
    try:
        result = await call_mcp_tool(
            'search_nodes', {'query': 'Alice', 'group_ids': [group_id], 'max_nodes': 5}
        )
        print(f'  Found: {json.dumps(result, indent=2, default=str)}')
    except Exception as e:
        print(f'  Error: {e}')

    # Search for TechCorp
    print("\nSearching for 'TechCorp'...")
    try:
        result = await call_mcp_tool(
            'search_nodes', {'query': 'TechCorp company', 'group_ids': [group_id], 'max_nodes': 5}
        )
        print(f'  Found: {json.dumps(result, indent=2, default=str)}')
    except Exception as e:
        print(f'  Error: {e}')

    print('\n' + '=' * 60)
    print('SEARCHING FOR FACTS (EDGES)')
    print('=' * 60)

    # Search for relationships
    print("\nSearching for 'works at' relationships...")
    try:
        result = await call_mcp_tool(
            'search_facts',
            {'query': 'who works at TechCorp', 'group_ids': [group_id], 'max_facts': 5},
        )
        print(f'  Found: {json.dumps(result, indent=2, default=str)}')
    except Exception as e:
        print(f'  Error: {e}')

    # Search for collaboration
    print("\nSearching for 'CloudSync' project relationships...")
    try:
        result = await call_mcp_tool(
            'search_facts',
            {'query': 'CloudSync project development', 'group_ids': [group_id], 'max_facts': 5},
        )
        print(f'  Found: {json.dumps(result, indent=2, default=str)}')
    except Exception as e:
        print(f'  Error: {e}')


async def main():
    print('=' * 60)
    print('GRAPHITI MCP SERVER TEST')
    print('=' * 60)
    print(f'Server URL: {MCP_SERVER_URL}')
    print()

    try:
        # Step 1: Add test memories
        group_id = await add_test_memories()

        # Step 2: Wait for async processing
        await wait_for_processing(20)

        # Step 3: Search and verify
        await search_and_verify(group_id)

        print('\n' + '=' * 60)
        print('TEST COMPLETE')
        print('=' * 60)
        print(f'\nTest group_id: {group_id}')
        print('If you see nodes and facts above, the Gemini embeddings are working!')

    except Exception as e:
        print(f'\nERROR: {e}')
        print('\nMake sure:')
        print('  1. The MCP server is running: uv run python src/graphiti_mcp_server.py')
        print('  2. FalkorDB is running: docker compose up -d')
        print('  3. The server is accessible at', MCP_SERVER_URL)
        raise


if __name__ == '__main__':
    asyncio.run(main())
