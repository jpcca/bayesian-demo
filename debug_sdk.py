#!/usr/bin/env python3
"""Debug script to see what Claude Agent SDK returns"""

import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def test_query():
    prompt = """You are a test system. Please respond with this exact JSON:
{
  "test": "success",
  "number": 42
}"""

    print("Sending prompt to Claude Agent SDK...")
    print("="*60)

    response_text = ""
    message_count = 0

    async for message in query(prompt=prompt, options=ClaudeAgentOptions()):
        message_count += 1
        print(f"\nMessage {message_count}:")
        print(f"  Type: {type(message)}")
        print(f"  Has 'type' attr: {hasattr(message, 'type')}")

        if hasattr(message, 'type'):
            print(f"  message.type: {message.type}")

        if hasattr(message, 'content'):
            print(f"  Has 'content': True")
            print(f"  Content type: {type(message.content)}")

            if isinstance(message.content, list):
                for i, block in enumerate(message.content):
                    print(f"    Block {i}: {type(block)}")
                    print(f"      Attributes: {dir(block)}")
                    if hasattr(block, 'text'):
                        print(f"      text: {block.text[:100]}...")
                        response_text += block.text
            else:
                print(f"    Content: {message.content}")

        # Print all attributes
        print(f"  All attributes: {[a for a in dir(message) if not a.startswith('_')]}")

    print("\n" + "="*60)
    print(f"Total messages received: {message_count}")
    print(f"Collected response_text ({len(response_text)} chars):")
    print(response_text)
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_query())
