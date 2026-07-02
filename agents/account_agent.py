import re

from agents.base import AgentResult
from agents.base import BaseAgent

SYSTEM_PROMPT = """
You are the ACCOUNT AGENT in a retail customer service system.
You handle: customer profile lookups only.

TOOLS YOU CAN USE:
- get_customer(customer_id)

NEVER invent customer names, IDs, or refund counts.

=== FLOW ===
1. If customer_id in context → call get_customer(customer_id)
2. If customer_id missing → action=needs_input, ask for it
3. Report what get_customer returns
4. If customer not found → say "Customer not found"

=== RESPONSE FORMAT ===
Raw JSON only.

{"action": "tool", "tool_calls": [{"tool": "get_customer", "arguments": {"customer_id": "..."}}]}
{"action": "needs_input", "summary": "", "prompt": "<question>"}
{"action": "done", "summary": "<factual summary>", "state_updates": {"customer_id": "..."}}

=== EXAMPLE ===

Task: "Look up customer usr_99"
{"action": "tool", "tool_calls": [{"tool": "get_customer", "arguments": {"customer_id": "usr_99"}}]}

Tool: "Customer ID: usr_99, Name: John Doe, Past Refunds: 0"
{"action": "done", "summary": "Customer usr_99 is John Doe with 0 past refunds.", "state_updates": {"customer_id": "usr_99"}}
"""

class AccountAgent(BaseAgent):
    name = "account_agent"
    allowed_tools = ["get_customer"]
    system_prompt = SYSTEM_PROMPT

    def _extract_customer_id(self, task_brief: str, shared_state: dict) -> str | None:
        match = re.search(r"\b(usr_\w+)\b", task_brief, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        if "customer_id" in shared_state:
            return str(shared_state["customer_id"]).lower()
        return None

    async def run(self, task_brief: str, shared_state: dict) -> AgentResult:
        tool_log = []
        customer_id = self._extract_customer_id(task_brief, shared_state)

        if not customer_id:
            return AgentResult(
                agent_name=self.name,
                status="needs_input",
                summary="Customer profile lookup needs a customer ID.",
                needs_input_prompt="Could you please provide your customer ID?",
                tool_log=tool_log,
            )

        result = await self._call_tool("get_customer", {"customer_id": customer_id})
        tool_log.append({
            "tool": "get_customer",
            "arguments": {"customer_id": customer_id},
            "result": result,
        })

        if result.startswith("Customer not found"):
            summary = f"No data found for customer ID {customer_id}."
        else:
            summary = result

        return AgentResult(
            agent_name=self.name,
            status="done",
            summary=summary,
            tool_log=tool_log,
            state_updates={"customer_id": customer_id},
        )
