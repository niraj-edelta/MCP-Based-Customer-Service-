import re

from agents.base import AgentResult
from agents.base import BaseAgent

SYSTEM_PROMPT = """
You are the REFUNDS AGENT in a retail customer service system.
You handle: refund eligibility checks and refund processing.
You do NOT handle order status or escalation to human.

TOOLS YOU CAN USE:
- check_refund_policy(order_id) → returns "ELIGIBLE" or "NOT_ELIGIBLE: <reason>"
- process_refund(order_id) → processes the refund (IRREVERSIBLE)
- get_order_amount(order_id) → returns order amount as string

NEVER call tools outside this list.
NEVER call process_refund without explicit customer confirmation.
NEVER invent amounts, statuses, or eligibility.

=== FLOW: REFUND REQUEST ===

Step 1 — Check eligibility:
- If order_id missing → action=needs_input, ask for it
- Call check_refund_policy(order_id)
- If NOT_ELIGIBLE: report reason, done. (Coordinator will offer escalation)
- If ELIGIBLE: proceed to Step 2



Step 3 — Check order amount:
- Call get_order_amount(order_id)
- If amount > 500: report that amount exceeds $500, escalation needed. Done.
  (Coordinator will escalate to human)
- If amount <= 500: report eligible and ask customer to confirm. Done.
  (Coordinator will ask customer to confirm)

Step 4 — Process refund (only when task brief explicitly says "confirmed"):
- Call process_refund(order_id)
- Report success

=== RESPONSE FORMAT ===
Raw JSON only. No markdown, no explanation.

Need a tool:
{"action": "tool", "tool_calls": [{"tool": "<name>", "arguments": {...}}]}

Need info from customer:
{"action": "needs_input", "summary": "<what you know>", "prompt": "<question>"}

Done:
{"action": "done", "summary": "<factual summary for coordinator>", "state_updates": {"order_id": "..."}}

=== EXAMPLES ===

Task: "Check refund eligibility for order 1234. Do not process yet."
{"action": "tool", "tool_calls": [{"tool": "check_refund_policy", "arguments": {"order_id": "1234"}}]}

Tool: "ELIGIBLE"
{"action": "tool", "tool_calls": [{"tool": "get_order_amount", "arguments": {"order_id": "1234"}}]}

Tool: "89.99"
{"action": "done", "summary": "Order 1234 is ELIGIBLE for a refund. Amount is $89.99 which is under $500. Ask customer to confirm they want to proceed with the refund.", "state_updates": {"order_id": "1234"}}

Task: "Check refund eligibility for order 9941. Do not process yet."
{"action": "tool", "tool_calls": [{"tool": "check_refund_policy", "arguments": {"order_id": "9941"}}]}

Tool: "NOT_ELIGIBLE: Purchase older than 30 days"
{"action": "done", "summary": "Order 9941 is NOT eligible for a refund because the purchase is older than 30 days (outside the 30-day window). Customer should be informed and offered escalation to human support.", "state_updates": {"order_id": "9941"}}

Task: "Check refund for order 9941. Amount is $650."
{"action": "tool", "tool_calls": [{"tool": "check_refund_policy", "arguments": {"order_id": "9941"}}]}

Tool: "ELIGIBLE"
{"action": "tool", "tool_calls": [{"tool": "get_order_amount", "arguments": {"order_id": "9941"}}]}

Tool: "650.0"
{"action": "done", "summary": "Order 9941 is ELIGIBLE for a refund. However the amount is $650.00 which exceeds $500. This must be escalated to human support instead of being processed automatically.", "state_updates": {"order_id": "9941"}}

Task: "Customer confirmed refund for order 1234. Eligibility already ELIGIBLE. Process refund now."
{"action": "tool", "tool_calls": [{"tool": "process_refund", "arguments": {"order_id": "1234"}}]}

Tool: "Refund successfully processed for order 1234"
{"action": "done", "summary": "Refund for order 1234 has been successfully processed.", "state_updates": {"order_id": "1234"}}
"""

class RefundsAgent(BaseAgent):
    name = "refunds_agent"
    allowed_tools = ["check_refund_policy", "process_refund", "get_order_amount"]
    system_prompt = SYSTEM_PROMPT

    def _extract_order_ids(self, text: str, shared_state: dict) -> list[str]:
        ids = []
        for match in re.findall(r"\bORD\s*-?\s*(\d{3,6})\b", text, re.IGNORECASE):
            ids.append(f"ORD{match}")
        for match in re.findall(r"\b(?<!ORD)(\d{3,6})\b", text, re.IGNORECASE):
            ids.append(f"ORD{match}")

        for order_id in shared_state.get("order_ids", []) or []:
            ids.append(str(order_id).upper())
        if "order_id" in shared_state:
            ids.append(str(shared_state["order_id"]).upper())

        normalized = []
        for order_id in ids:
            order_id = str(order_id).strip().upper()
            if re.fullmatch(r"\d{3,6}", order_id):
                order_id = f"ORD{order_id}"
            normalized.append(order_id)
        return list(dict.fromkeys(normalized))

    async def _check_one(self, order_id: str, tool_log: list) -> tuple[str, dict]:
        policy = await self._call_tool("check_refund_policy", {"order_id": order_id})
        tool_log.append({
            "tool": "check_refund_policy",
            "arguments": {"order_id": order_id},
            "result": policy,
        })

        if policy.startswith("NOT_ELIGIBLE"):
            reason = policy.split(":", 1)[1].strip() if ":" in policy else policy
            return f"{order_id}: Not eligible for refund - {reason}.", {"order_id": order_id}

        if policy != "ELIGIBLE":
            return f"{order_id}: {policy}.", {"order_id": order_id}

        amount = await self._call_tool("get_order_amount", {"order_id": order_id})
        tool_log.append({
            "tool": "get_order_amount",
            "arguments": {"order_id": order_id},
            "result": amount,
        })

        try:
            amount_value = float(amount)
        except ValueError:
            return f"{order_id}: Eligible for refund, but I could not verify the amount ({amount}).", {"order_id": order_id}

        if amount_value > 500:
            return (
                f"{order_id}: Eligible by policy, but the ${amount_value:.2f} amount exceeds "
                "$500 and needs human approval."
            ), {"order_id": order_id}

        return (
            f"{order_id}: Eligible for refund for ${amount_value:.2f}. "
            "Please confirm if you want this refund processed."
        ), {"order_id": order_id}

    async def run(self, task_brief: str, shared_state: dict) -> AgentResult:
        tool_log = []
        order_ids = self._extract_order_ids(task_brief, shared_state)

        if not order_ids:
            return AgentResult(
                agent_name=self.name,
                status="needs_input",
                summary="No order ID available for refund eligibility.",
                needs_input_prompt="Could you please provide the order ID you want checked for a refund?",
                tool_log=tool_log,
            )

        if "confirmed" in task_brief.lower() and len(order_ids) == 1:
            order_id = order_ids[0]
            result = await self._call_tool("process_refund", {"order_id": order_id})
            tool_log.append({
                "tool": "process_refund",
                "arguments": {"order_id": order_id},
                "result": result,
            })
            return AgentResult(
                agent_name=self.name,
                status="done",
                summary=result,
                tool_log=tool_log,
                state_updates={"order_id": order_id, "order_ids": order_ids},
            )

        summaries = []
        for order_id in order_ids:
            summary, _ = await self._check_one(order_id, tool_log)
            summaries.append(summary)

        return AgentResult(
            agent_name=self.name,
            status="done",
            summary="Refund eligibility: " + " ".join(summaries),
            tool_log=tool_log,
            state_updates={"order_id": order_ids[0], "order_ids": order_ids},
        )
