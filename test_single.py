#!/usr/bin/env python3
"""Test single prediction to verify the SDK is working"""

import asyncio
import json
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage

async def test():
    prompt = """You are a height and weight prediction system.
Given a text description of a person, predict their height (cm) and weight (kg)
as probability distributions (NOT point estimates).

Output JSON format:
{
  "reasoning": "explanation",
  "height_distribution": {"distribution_type": "normal", "mu": 175, "sigma": 6, "unit": "cm"},
  "weight_distribution": {"distribution_type": "normal", "mu": 70, "sigma": 8, "unit": "kg"}
}

USER INPUT:
Sarah is a 32-year-old Norwegian woman who works as a software engineer. She mentioned playing volleyball in college and still plays recreationally on weekends. She describes herself as taller than most of her female friends and maintains an active lifestyle with regular gym sessions.

Please respond with ONLY the JSON object, no additional text."""

    print("Sending query...")
    response_text = ""

    async for message in query(prompt=prompt, options=ClaudeAgentOptions()):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    response_text += block.text
                    print(f"Got text: {block.text[:100]}...")
        elif isinstance(message, ResultMessage):
            print(f"Result: {message.subtype}, duration: {message.duration_ms}ms")

    print("\n" + "="*60)
    print("Full response:")
    print(response_text)
    print("="*60)

    # Try to parse
    try:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        data = json.loads(response_text.strip())
        print("\nParsed successfully:")
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"\nParsing failed: {e}")

if __name__ == "__main__":
    asyncio.run(test())
