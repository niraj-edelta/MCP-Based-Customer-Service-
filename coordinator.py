import json
import re
from dataclasses import dataclass

from ollama import chat

from agents.json_utils import safe_json_loads

MODEL_NAME = "llama3.2:3b"


# ── Python-level ID extraction (reliable, no LLM) ─────────────────────────────
def _extract_ids(message: str, known: dict) -> dict:
    updates = {}
    order_ids = _extract_order_ids(message)
    if order_ids:
        existing = known.get("order_ids", [])
        merged = list(dict.fromkeys([*existing, *order_ids]))
        updates["order_ids"] = merged
        updates["order_id"] = order_ids[0]
    # Order IDs: 3-6 digit numbers
    if "order_id" not in known and "order_id" not in updates:
        m = re.findall(r'\b(\d{3,6})\b', message)
        if m:
            updates["order_id"] = _normalize_order_id(m[0])
    # Customer IDs: usr_XX
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


def _normalize_order_id(order_id: str) -> str:
    order_id = str(order_id).strip().upper()
    if re.fullmatch(r"\d{3,6}", order_id):
        return f"ORD{order_id}"
    return order_id


def _extract_order_ids(text: str) -> list[str]:
    ids = []
    for match in re.findall(r"\bORD\s*-?\s*(\d{3,6})\b", text, re.IGNORECASE):
        ids.append(f"ORD{match}")
    for match in re.findall(r"\b(?<!ORD)(\d{3,6})\b", text, re.IGNORECASE):
        ids.append(f"ORD{match}")
    return list(dict.fromkeys(ids))


def _extract_order_ids_from_results(agent_results: list) -> list[str]:
    ids = []
    for result in agent_results:
        ids.extend(_extract_order_ids(result.summary or ""))
        for tool_entry in result.tool_log:
            ids.extend(_extract_order_ids(tool_entry.get("result", "")))
            args = tool_entry.get("arguments", {}) or {}
            if "order_id" in args:
                ids.append(_normalize_order_id(args["order_id"]))
    return list(dict.fromkeys(ids))


def _last_customer_message(conversation_log: str) -> str:
    matches = re.findall(r"^Customer:\s*(.+)$", conversation_log or "", re.MULTILINE)
    return matches[-1] if matches else ""


def _recent_text(conversation_log: str) -> str:
    return (conversation_log or "").lower()[-1200:]


def _recent_asked_for_customer_id(conversation_log: str) -> bool:
    recent = _recent_text(conversation_log)
    return "customer id" in recent or "customer_id" in recent


def _recent_asked_for_order_id(conversation_log: str) -> bool:
    recent = _recent_text(conversation_log)
    return "order id" in recent or "order_id" in recent


def _single_token_id_candidate(message: str) -> str | None:
    msg = message.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,40}", msg):
        return None
    if msg.lower() in set(GREETINGS + ["yes", "no", "all", "ok", "okay"]):
        return None
    return msg


def _asks_capabilities(message: str) -> bool:
    msg = message.lower().strip()
    patterns = [
        "what you can do",
        "what can you do",
        "how can you help",
        "what do you do",
        "help me with",
    ]
    return any(pattern in msg for pattern in patterns)


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
META_KW       = ["last question", "previous question", "what did i ask",
                 "what was my question"]


def _detect_intent(message: str) -> str:
    msg = message.lower().strip()
    for kw in META_KW:
        if kw in msg:
            return "meta"
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


def _wants_all_relevant_orders(message: str) -> bool:
    msg = message.lower()
    return any(token in msg for token in ["all", "all three", "these three", "those three", "which one", "which order"])


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
Write ONE clear, friendly, natural reply to the customer.

STRICT RULES:
- Use ONLY facts from the provided fact payload. Never invent facts.
- Preserve exact order IDs, item names, amounts, and refund eligibility.
- If tool results say "Customer not found", "Order not found", or "No orders found", clearly say no data was found.
- If an order is marked not eligible, do not call it eligible.
- If an order is marked already refunded, say it is not eligible because it was already refunded.
- If summary says "ask customer to confirm" → include that question.
- If summary says "escalate to human" → tell customer you are escalating.
- If summary says "ask which order" → ask the customer which order.
- If status is "needs_input" → ask for that missing info.
- Plain text only. No JSON, no bullet points, no agent names,
  no markdown, no internal terms like "orders_agent".

Respond with ONLY the plain text reply.
"""


@dataclass
class TurnData:
    reply_context: str
    shared_state: dict
    agent_results: list


class Coordinator:
    def __init__(self, agents: dict):
        self.agents = agents

    def _deterministic_route(self, conversation_log: str, shared_state: dict, customer_message: str) -> dict | None:
        msg = customer_message.lower().strip()
        id_candidate = _single_token_id_candidate(customer_message)

        if id_candidate and _recent_asked_for_customer_id(conversation_log):
            return {
                "action": "dispatch",
                "dispatch": [{
                    "agent": "orders_agent",
                    "brief": f"Get full order history for customer {id_candidate.lower()}.",
                }],
            }

        if id_candidate and _recent_asked_for_order_id(conversation_log):
            return {
                "action": "dispatch",
                "dispatch": [{
                    "agent": "orders_agent",
                    "brief": f"Look up order details for {_normalize_order_id(id_candidate)}.",
                }],
            }

        if any(kw in msg for kw in SUPPORT_KW):
            return {
                "action": "dispatch",
                "dispatch": [{
                    "agent": "support_agent",
                    "brief": "Customer explicitly asked to speak to a human agent. Escalate immediately.",
                }],
            }

        current_order_ids = _extract_order_ids(customer_message)
        known_order_ids = shared_state.get("order_ids", [])
        if "order_id" in shared_state and shared_state["order_id"] not in known_order_ids:
            known_order_ids = [*known_order_ids, shared_state["order_id"]]

        is_refund_continuation = (
            msg in {"all", "all of them", "these", "these three", "those", "those three"}
            and any(kw in (conversation_log or "").lower()[-1000:] for kw in REFUND_KW)
        )

        if any(kw in msg for kw in REFUND_KW) or is_refund_continuation:
            if current_order_ids:
                order_ids = current_order_ids
            elif _wants_all_relevant_orders(customer_message) and known_order_ids:
                order_ids = known_order_ids
            elif "order_id" in shared_state:
                order_ids = [shared_state["order_id"]]
            else:
                order_ids = []

            if order_ids:
                return {
                    "action": "dispatch",
                    "dispatch": [{
                        "agent": "refunds_agent",
                        "brief": (
                            "Check refund eligibility for these orders without processing: "
                            + ", ".join(order_ids)
                            + ". Report every order separately using only tool results."
                        ),
                    }],
                }

            if "customer_id" in shared_state and _wants_all_relevant_orders(customer_message):
                return {
                    "action": "dispatch",
                    "dispatch": [{
                        "agent": "orders_agent",
                        "brief": (
                            f"Get full order history for customer {shared_state['customer_id']} "
                            "so refund eligibility can be checked for all returned orders."
                        ),
                    }],
                }

            return {
                "action": "dispatch",
                "dispatch": [{
                    "agent": "refunds_agent",
                    "brief": "Customer asked about refund eligibility but no order_id is known. Ask for the order ID.",
                }],
            }

        if any(kw in msg for kw in ORDER_KW):
            if "history" in msg or "previous" in msg or "order list" in msg or "my orders" in msg or "product" in msg:
                if "customer_id" in shared_state:
                    return {
                        "action": "dispatch",
                        "dispatch": [{
                            "agent": "orders_agent",
                            "brief": f"Get full order history for customer {shared_state['customer_id']}.",
                        }],
                    }
                return {
                    "action": "dispatch",
                    "dispatch": [{
                        "agent": "orders_agent",
                        "brief": "Customer wants order history but customer_id is missing. Ask for customer_id.",
                    }],
                }

            if current_order_ids or "order_id" in shared_state:
                order_id = (current_order_ids or [shared_state["order_id"]])[0]
                return {
                    "action": "dispatch",
                    "dispatch": [{
                        "agent": "orders_agent",
                        "brief": f"Look up order details for {order_id}.",
                    }],
                }

        if any(kw in msg for kw in ACCOUNT_KW):
            if "customer_id" in shared_state:
                return {
                    "action": "dispatch",
                    "dispatch": [{
                        "agent": "account_agent",
                        "brief": f"Look up customer {shared_state['customer_id']}.",
                    }],
                }

        return None

    def _route(self, conversation_log: str, shared_state: dict, customer_message: str) -> dict:
        deterministic = self._deterministic_route(conversation_log, shared_state, customer_message)
        if deterministic:
            return deterministic

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

    def _synthesis_messages(
        self,
        customer_message: str,
        agent_results: list,
        reply_context: str = "",
    ) -> list:
        results_payload = [
            {
                "agent": r.agent_name,
                "status": r.status,
                "summary": r.summary,
                "needs_input_prompt": r.needs_input_prompt,
                "tool_results": [
                    {
                        "tool": t.get("tool"),
                        "arguments": t.get("arguments"),
                        "result": t.get("result"),
                    }
                    for t in r.tool_log
                ],
            }
            for r in agent_results
        ]

        return [
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Customer message: {customer_message}\n\n"
                    f"Direct reply context:\n{reply_context or 'None'}\n\n"
                    f"Agent results:\n{json.dumps(results_payload, indent=2)}\n\n"
                    "Write the customer-facing reply now. Make it sound natural, "
                    "but do not change any facts."
                ),
            },
        ]

    def _synthesize(self, customer_message: str, agent_results: list, reply_context: str = "") -> str:
        messages = self._synthesis_messages(customer_message, agent_results, reply_context)
        response = chat(model=MODEL_NAME, messages=messages)
        return response["message"]["content"].strip()

    def synthesize_stream(self, customer_message: str, agent_results: list, reply_context: str = ""):
        messages = self._synthesis_messages(customer_message, agent_results, reply_context)
        for chunk in chat(model=MODEL_NAME, messages=messages, stream=True):
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    async def prepare_turn(self, conversation_log: str, shared_state: dict, customer_message: str) -> TurnData:
        # Always extract IDs first in Python
        id_updates = _extract_ids(customer_message, shared_state)
        shared_state = {**shared_state, **id_updates}

        id_candidate = _single_token_id_candidate(customer_message)
        if id_candidate and _recent_asked_for_customer_id(conversation_log):
            shared_state["customer_id"] = id_candidate.lower()
        elif id_candidate and _recent_asked_for_order_id(conversation_log):
            shared_state["order_id"] = _normalize_order_id(id_candidate)

        intent = _detect_intent(customer_message)

        if intent == "meta":
            last_question = _last_customer_message(conversation_log)
            context = (
                f"The customer's previous message was: {last_question}"
                if last_question else
                "There is no previous customer message in this chat yet."
            )
            return TurnData(context, shared_state, [])

        if intent == "greeting":
            return TurnData(
                "The customer greeted the assistant. Reply briefly and ask how you can help with orders, refunds, or their account.",
                shared_state,
                [],
            )

        if intent == "offtopic":
            return TurnData(
                "The customer asked something outside retail support. Politely say you can help with orders, refunds, or account questions.",
                shared_state,
                [],
            )

        if _asks_capabilities(customer_message):
            return TurnData(
                "Explain that you can help with order history, order details/status, refund eligibility, refund processing after confirmation, reorders, account lookups, and human support escalation. Ask what they would like help with.",
                shared_state,
                [],
            )

        # Route to agents
        routing = self._route(conversation_log, shared_state, customer_message)

        if routing.get("action") == "reply_directly":
            return TurnData(
                f"Coordinator direct reply instruction: {routing.get('message', '')}",
                shared_state,
                [],
            )

        dispatch_list = routing.get("dispatch", [])
        if not dispatch_list:
            return TurnData(
                "The user request is unclear. Ask one concise clarifying question.",
                shared_state,
                [],
            )

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

        found_order_ids = _extract_order_ids_from_results(agent_results)
        if found_order_ids:
            current_order_ids = updated_state.get("order_ids", [])
            updated_state["order_ids"] = list(dict.fromkeys([*current_order_ids, *found_order_ids]))
            updated_state.setdefault("order_id", found_order_ids[0])

        if (
            len(agent_results) == 1
            and agent_results[0].agent_name == "orders_agent"
            and any(kw in customer_message.lower() for kw in REFUND_KW)
        ):
            order_ids = updated_state.get("order_ids", [])
            if order_ids:
                result = await self.agents["refunds_agent"].run(
                    "Check refund eligibility for these orders without processing: "
                    + ", ".join(order_ids)
                    + ". Report every order separately using only tool results.",
                    updated_state,
                )
                agent_results.append(result)
                updated_state.update(result.state_updates)

        return TurnData("", updated_state, agent_results)

    async def handle_turn(self, conversation_log: str, shared_state: dict, customer_message: str):
        turn = await self.prepare_turn(conversation_log, shared_state, customer_message)
        reply_text = self._synthesize(customer_message, turn.agent_results, turn.reply_context)
        return reply_text, turn.shared_state, turn.agent_results
