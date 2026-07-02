import re

from agents.base import AgentResult
from agents.base import BaseAgent

SYSTEM_PROMPT = """
You are the ORDERS AGENT in a retail customer service system.
You handle: order status/details, order history, and reorders.
You do NOT handle refunds or escalation.

TOOLS YOU CAN USE:
- lookup_order(order_id)
- get_order_history(customer_id)
- create_reorder(order_id, quantity)
- get_order_amount(order_id)

NEVER call tools outside this list.
NEVER invent or guess any order IDs, amounts, or statuses.
NEVER call create_reorder without explicit customer confirmation.

=== FLOW 1: ORDER DETAILS ===
Customer asks: order status, where is my order, track order
1. If order_id is in context → call lookup_order(order_id)
2. If order_id is missing → action=needs_input, ask for order_id
3. If order found → report details
4. If not found → say "Order not found"

=== FLOW 2: ORDER HISTORY ===
Customer asks: order history, previous orders, show my orders
1. If customer_id is in context → call get_order_history(customer_id)
2. If customer_id is missing → action=needs_input, ask for customer_id
3. Show ALL orders returned

=== FLOW 3: REORDER ===
Customer asks: reorder, order again, buy again
Step A — Get history first:
1. If customer_id in context → call get_order_history(customer_id)
2. If customer_id missing → action=needs_input, ask for customer_id
3. Show all orders to customer

Step B — After showing history (task brief will say "ask which order"):
- If only 1 order: show it and ask "Would you like to reorder this?"
- If multiple orders: list them all and ask "Which order would you like to reorder?"
- Do NOT call create_reorder yet

Step C — After customer confirms specific order (task brief will say "confirmed"):
- Call create_reorder(order_id, quantity)
- Default quantity = 1 unless customer specified

=== RESPONSE FORMAT ===
You respond ONLY with raw JSON. No markdown, no explanation.

Need a tool:
{"action": "tool", "tool_calls": [{"tool": "<name>", "arguments": {...}}]}

Need info from customer:
{"action": "needs_input", "summary": "<what you know>", "prompt": "<question for customer>"}

Done:
{"action": "done", "summary": "<factual summary for coordinator>", "state_updates": {"order_id": "...", "customer_id": "..."}}

=== EXAMPLES ===

Task: "Look up status of order 1234"
{"action": "tool", "tool_calls": [{"tool": "lookup_order", "arguments": {"order_id": "1234"}}]}

Tool result: "Order ID: 1234, Item: Wireless Headphones, Status: Shipped - Arriving Tomorrow, Amount: $89.99"
{"action": "done", "summary": "Order 1234 (Wireless Headphones, $89.99) has shipped and is arriving tomorrow.", "state_updates": {"order_id": "1234"}}

Task: "Get order history for customer usr_99"
{"action": "tool", "tool_calls": [{"tool": "get_order_history", "arguments": {"customer_id": "usr_99"}}]}

Tool result: "Order ID: 1234, Item: Wireless Headphones, Status: Shipped, Amount: $89.99\nOrder ID: 5678, Item: Gaming Mouse, Status: Delivered, Amount: $49.99"
{"action": "done", "summary": "Customer usr_99 has 2 orders: Order 1234 (Wireless Headphones, $89.99, Shipped) and Order 5678 (Gaming Mouse, $49.99, Delivered). Ask which one they want to reorder.", "state_updates": {"customer_id": "usr_99"}}

Task: "Customer confirmed reorder of order 5678 quantity 1. Call create_reorder now."
{"action": "tool", "tool_calls": [{"tool": "create_reorder", "arguments": {"order_id": "5678", "quantity": 1}}]}

Tool result: "Reorder created successfully. Item: Gaming Mouse, Quantity: 1, Total Amount: $49.99"
{"action": "done", "summary": "Reorder placed successfully: 1x Gaming Mouse for $49.99.", "state_updates": {"order_id": "5678"}}

Task: "Look up order but order_id missing"
{"action": "needs_input", "summary": "", "prompt": "Could you please provide your order ID?"}
"""

class OrdersAgent(BaseAgent):
    name = "orders_agent"
    allowed_tools = ["lookup_order", "get_order_history", "create_reorder", "get_order_amount"]
    system_prompt = SYSTEM_PROMPT

    def _normalize_order_id(self, order_id: str) -> str:
        order_id = str(order_id).strip().upper()
        if re.fullmatch(r"\d{3,6}", order_id):
            return f"ORD{order_id}"
        return order_id

    def _extract_order_ids(self, text: str, shared_state: dict) -> list[str]:
        ids = []
        for match in re.findall(r"\bORD\s*-?\s*(\d{3,6})\b", text, re.IGNORECASE):
            ids.append(f"ORD{match}")
        for match in re.findall(r"\b(?<!ORD)(\d{3,6})\b", text, re.IGNORECASE):
            ids.append(f"ORD{match}")
        if "order_id" in shared_state:
            ids.append(shared_state["order_id"])
        return list(dict.fromkeys(self._normalize_order_id(order_id) for order_id in ids))

    def _format_history(self, result_text: str, include_reorder_question: bool) -> str:
        if result_text.startswith("No orders found"):
            customer_id = result_text.replace("No orders found for customer", "").strip()
            if customer_id:
                return f"No data found for customer ID {customer_id}."
            return "No data found for that customer ID."

        lines = [line.strip() for line in result_text.splitlines() if line.strip()]
        if include_reorder_question:
            return "Here are your previous orders:\n" + "\n".join(lines) + "\nWhich order would you like to reorder?"
        return "Your order history is as follows:\n" + "\n".join(lines)

    async def run(self, task_brief: str, shared_state: dict) -> AgentResult:
        brief_lower = task_brief.lower()
        tool_log = []

        is_reorder = "reorder" in brief_lower or "order again" in brief_lower or "buy again" in brief_lower
        wants_history = (
            "history" in brief_lower
            or "previous orders" in brief_lower
            or "all orders" in brief_lower
            or "full order" in brief_lower
            or "customer" in brief_lower and "order" in brief_lower
        )

        customer_match = re.search(r"\b(usr_\w+)\b", task_brief, re.IGNORECASE)
        customer_id = (
            customer_match.group(1).lower()
            if customer_match else
            shared_state.get("customer_id")
        )

        if wants_history and customer_id:
            result = await self._call_tool("get_order_history", {"customer_id": customer_id})
            tool_log.append({
                "tool": "get_order_history",
                "arguments": {"customer_id": customer_id},
                "result": result,
            })
            order_ids = []
            for match in re.findall(r"\bORD\s*-?\s*(\d{3,6})\b", result, re.IGNORECASE):
                order_ids.append(f"ORD{match}")
            return AgentResult(
                agent_name=self.name,
                status="done",
                summary=self._format_history(result, is_reorder),
                tool_log=tool_log,
                state_updates={"customer_id": customer_id, "order_ids": list(dict.fromkeys(order_ids))},
            )

        if wants_history and "customer_id" not in shared_state:
            return AgentResult(
                agent_name=self.name,
                status="needs_input",
                summary="Customer wants order history but customer_id is missing.",
                needs_input_prompt="Could you please provide your customer ID?",
                tool_log=tool_log,
            )

        if "details" in brief_lower or "status" in brief_lower or "look up order" in brief_lower:
            order_ids = self._extract_order_ids(task_brief, shared_state)
            if not order_ids:
                return AgentResult(
                    agent_name=self.name,
                    status="needs_input",
                    summary="Customer wants order details but order_id is missing.",
                    needs_input_prompt="Could you please provide your order ID?",
                    tool_log=tool_log,
                )

            order_id = order_ids[0]
            result = await self._call_tool("lookup_order", {"order_id": order_id})
            tool_log.append({
                "tool": "lookup_order",
                "arguments": {"order_id": order_id},
                "result": result,
            })
            summary = (
                f"No data found for order ID {order_id}."
                if result.startswith("Order not found") else
                result
            )
            return AgentResult(
                agent_name=self.name,
                status="done",
                summary=summary,
                tool_log=tool_log,
                state_updates={"order_id": order_id, "order_ids": order_ids},
            )

        return await super().run(task_brief, shared_state)
