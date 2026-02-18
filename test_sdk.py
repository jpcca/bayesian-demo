"""Quick SDK connectivity test — sends a single minimal request."""

import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage


async def main():
    options = ClaudeAgentOptions(model="haiku", max_turns=1, max_thinking_tokens=0)
    print("Sending test request to Claude via SDK...")
    async for message in query(prompt="Reply with just: OK", options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    print(f"Response: {block.text.strip()}")
        elif isinstance(message, ResultMessage):
            print(f"Done. is_error={message.is_error}, turns={message.num_turns}")
            if message.usage:
                print(f"Tokens: {message.usage}")


asyncio.run(main())
