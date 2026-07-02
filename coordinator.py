import json
import re

from ollama import chat

from agents.json_utils import safe_json_loads

MODEL_NAME = "llama3.2:3b"


# ── Python-level ID extraction (reliable, no LLM) ─────────────────────────────
def _extract_ids(message: str, known: dict) -> dict:
    updates = {}
    # Order IDs: 3-6 digit numbers
    if "order_id" not in known:
        m = re.findall(r'\b(\d{3,6})\b', message)
        if m:
            updates["order_id"] = m[0]
    # Customer IDs: usr_XX
    if "customer_id" not in known:
        m = re.findall(r'\b(usr_\w+)\b', message, re.IGNORECASE)
        if m:
            updates["customer_id"] = m[0].lower()
    # Name → customer_id mapping
    NAMES = {
        "john doe": "usr_99", "john": "usr_99",
        "jane smith": "usr_100", "jane": "usr_100",
    }
    if "customer_id" not in known and "customer_id" not in updates:
        msg_lower = message.lower()
        for name, cid in NAMES.items():
            if name in msg_lower:
                updates["customer_id"] = cid
                break
    return updates


# ── Python-level intent detection ─────────────────────────────────────────────
ORDER_KW      = ["order", "status", "track", "where is", "shipped", "delivery",
                 "deliver", "history", "previous", "reorder", "order again",
                 "buy again", "item", "package", "order list", "my order"]
REFUND_KW     = ["refund", "return", "money back", "cancel", "eligible",
                 "refundable", "get my money"]
ACCOUNT_KW    = ["account", "profile", "customer", "my info", "customer id",
                 "customer list"]
SUPPORT_KW    = ["human", "agent", "support", "speak to", "talk to", "person",
                 "representative", "escalate"]
GREETINGS     = ["hi", "hello", "hey", "thanks", "thank you", "ok",
                 "okay", "bye", "goodbye", "good morning", "good afternoon"]
OFFTOPIC_KW   = ["who is", "capital of", "weather", "temperature", "joke",
                 "prime minister", "president", "cricket", "football",
                 "movie", "recipe", "write code", "translate"]


def _detect_intent(message: str) -> str:
    msg = message.lower().strip()
    # Short messages (≤3 words) are always retail context
    if len(msg.split()) <= 3:
        return "retail"
    # Retail keyword match
    for kw in ORDER_KW + REFUND_KW + ACCOUNT_KW + SUPPORT_KW:
        if kw in msg:
            return "retail"
    # Greeting
    for g in GREETINGS:
        if msg == g or msg.startswith(g + " "):
            return "greeting"
    # Off-topic
    for ot in OFFTOPIC_KW:
        if ot in msg:
            return "offtopic"
    return "retail"  # default: assume retail


# ── Routing prompt ─────────────────────────────────────────────────────────────
ROUTING_PROMPT = """
You are the COORDINATOR of a multi-agent retail customer service system.
Route the customer message to the correct specialist agent(s).

AGENTS:
- orders_agent  → order status/details, order history, reorders
- refunds_agent → refund eligibility check, refund processing
- account_agent → customer profile lookup
- support_agent → escalate to human

ROUTING RULES:
1. Always include all known IDs from shared_state in the agent brief.
2. Short replies like "yes", "no", "1234", "usr_100" are continuations
   — read the recent conversation and route accordingly.
3. If customer confirms reorder → brief orders_agent with confirmed order_id and quantity.
4. If customer confirms refund → brief refunds_agent saying confirmed, process it.
5. If customer wants multiple things → dispatch multiple agents at once.
6. If an ID is missing, still dispatch the agent — it will ask.

Respond ONLY with raw JSON, no explanation, no markdown.

{"action": "dispatch", "dispatch": [{"agent": "<name>", "brief": "<full task with all known IDs>"}]}

EXAMPLES:

Shared: {} | Customer: "where is my order 1234"
{"action": "dispatch", "dispatch": [{"agent": "orders_agent", "brief": "Look up status and details of order 1234."}]}

Shared: {"customer_id": "usr_99"} | Customer: "show my order history"
{"action": "dispatch", "dispatch": [{"agent": "orders_agent", "brief": "Get full order history for customer usr_99."}]}

Shared: {"customer_id": "usr_99"} | Customer: "I want to reorder"
{"action": "dispatch", "dispatch": [{"agent": "orders_agent", "brief": "Customer usr_99 wants to reorder. Get their order history and show all orders. If only one order, ask to confirm. If multiple, ask which one they want to reorder."}]}

Shared: {"customer_id": "usr_99", "order_id": "5678"} | Recent: agent showed orders and asked which to reorder | Customer: "the gaming mouse"
{"action": "dispatch", "dispatch": [{"agent": "orders_agent", "brief": "Customer confirmed they want to reorder order 5678 (Gaming Mouse). Ask for quantity if not given, then confirm before placing."}]}

Shared: {"customer_id": "usr_99", "order_id": "5678"} | Recent: asked to confirm reorder of 5678 qty 1 | Customer: "yes"
{"action": "dispatch", "dispatch": [{"agent": "orders_agent", "brief": "Customer confirmed reorder of order 5678 quantity 1. Call create_reorder now."}]}

Shared: {"order_id": "1234"} | Customer: "can I get a refund"
{"action": "dispatch", "dispatch": [{"agent": "refunds_agent", "brief": "Check refund eligibility for order 1234. Do NOT process yet, just check and report."}]}

Shared: {"order_id": "1234"} | Recent: refunds_agent said eligible, asked to confirm | Customer: "yes"
{"action": "dispatch", "dispatch": [{"agent": "refunds_agent", "brief": "Customer confirmed refund for order 1234. Eligibility already confirmed ELIGIBLE. Check amount — if >$500 escalate, if <=$500 process refund now."}]}

Shared: {"order_id": "9941"} | Customer: "refund my order"
{"action": "dispatch", "dispatch": [{"agent": "refunds_agent", "brief": "Customer wants refund for order 9941. Check eligibility first. If not eligible, inform and explain. If eligible, check amount — >$500 escalate to human, <=$500 ask confirmation then process."}]}

Shared: {} | Customer: "talk to a human"
{"action": "dispatch", "dispatch": [{"agent": "support_agent", "brief": "Customer explicitly asked to speak to a human agent. Escalate immediately."}]}

Shared: {"order_id": "1234"} | Customer: "status and refund for order 1234"
{"action": "dispatch", "dispatch": [
  {"agent": "orders_agent", "brief": "Look up status of order 1234."},
  {"agent": "refunds_agent", "brief": "Check refund eligibility for order 1234. Do not process yet."}
]}
"""

SYNTHESIS_PROMPT = """
You are a retail customer service coordinator.
One or more specialist agents completed tasks. Write ONE clear,
friendly, natural reply to the customer combining all results.

STRICT RULES:
- Use ONLY facts from agent summaries. Never invent facts.
- If summary says "ask customer to confirm" → include that question.
- If summary says "escalate to human" → tell customer you are escalating.
- If summary says "ask which order" → ask the customer which order.
- If status is "needs_input" → ask for that missing info.
- Plain text only. No JSON, no bullet points, no agent names,
  no markdown, no internal terms like "orders_agent".

Respond with ONLY the plain text reply.
"""


class Coordinator:
    def __init__(self, agents: dict):
        self.agents = agents

    def _route(self, conversation_log: str, shared_state: dict, customer_message: str) -> dict:
        messages = [
            {"role": "system", "content": ROUTING_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Shared state (confirmed IDs): {json.dumps(shared_state)}\n"
                    f"Recent conversation:\n{conversation_log[-2000:] if conversation_log else 'None'}\n"
                    f"Customer: {customer_message}\n\n"
                    "Respond ONLY with JSON dispatch."
                ),
            },
        ]
        response = chat(model=MODEL_NAME, messages=messages, format="json")
        parsed = safe_json_loads(response["message"]["content"])
        if parsed is None:
            return {
                "action": "dispatch",
                "dispatch": [{"agent": "orders_agent",
                               "brief": f"Customer said: '{customer_message}'. Known: {json.dumps(shared_state)}. Help them."}],
            }
        return parsed

    def _synthesize(self, customer_message: str, agent_results: list) -> str:
        results_payload = [
            {
                "agent": r.agent_name,
                "status": r.status,
                "summary": r.summary,
                "needs_input_prompt": r.needs_input_prompt,
            }
            for r in agent_results
        ]
        messages = [
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Customer message: {customer_message}\n\n"
                    f"Agent results:\n{json.dumps(results_payload, indent=2)}\n\n"
                    "Write the plain text reply now."
                ),
            },
        ]
        response = chat(model=MODEL_NAME, messages=messages)
        return response["message"]["content"].strip()

    async def handle_turn(self, conversation_log: str, shared_state: dict, customer_message: str):
        # Always extract IDs first in Python
        id_updates = _extract_ids(customer_message, shared_state)
        shared_state = {**shared_state, **id_updates}

        intent = _detect_intent(customer_message)

        if intent == "greeting":
            replies = {
                "hi": "Hello! How can I help you with your order today?",
                "hello": "Hello! How can I help you today?",
                "hey": "Hey! How can I help you?",
                "thanks": "You're welcome! Let me know if there's anything else.",
                "thank you": "You're welcome! Is there anything else I can help with?",
                "bye": "Goodbye! Have a great day!",
                "goodbye": "Goodbye! Have a great day!",
            }
            reply = replies.get(customer_message.lower().strip(),
                                "Hello! How can I help you with your order today?")
            return reply, shared_state, []

        if intent == "offtopic":
            return (
                "I can only help with order and retail related questions. "
                "How can I assist you with your orders or account?",
                shared_state, []
            )

        # Route to agents
        routing = self._route(conversation_log, shared_state, customer_message)

        if routing.get("action") == "reply_directly":
            return routing.get("message", ""), shared_state, []

        dispatch_list = routing.get("dispatch", [])
        if not dispatch_list:
            return "Could you tell me more about what you need help with?", shared_state, []

        agent_results = []
        updated_state = dict(shared_state)

        for item in dispatch_list:
            agent_name = item.get("agent")
            brief = item.get("brief", "")
            agent = self.agents.get(agent_name)
            if agent is None:
                continue
            result = await agent.run(brief, updated_state)
            agent_results.append(result)
            updated_state.update(result.state_updates)

        reply_text = self._synthesize(customer_message, agent_results)
        return reply_text, updated_state, agent_results