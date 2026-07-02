import json
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ollama import chat as ollama_chat

sys.path.insert(0, os.path.dirname(__file__))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.orders_agent import OrdersAgent
from agents.refunds_agent import RefundsAgent
from agents.account_agent import AccountAgent
from agents.support_agent import SupportAgent
from coordinator import Coordinator, _detect_intent, _extract_ids
from agents.json_utils import safe_json_loads

MODEL_NAME = "llama3.2:3b"

SINGLE_AGENT_PROMPT = """
You are a Customer Service Agent for an online retail store.

TOOLS:
1. lookup_order(order_id)
2. get_order_history(customer_id)
3. create_reorder(order_id, quantity)
4. check_refund_policy(order_id)
5. process_refund(order_id)
6. get_customer(customer_id)
7. get_order_amount(order_id)
8. escalate_to_human(reason)

IMPORTANT: The user message may contain [Known IDs: {...}].
Always use those IDs directly. Never ask for an ID that is already known.
Never invent IDs, amounts, names, or statuses.

=== FLOW 1: ORDER DETAILS ===
- If order_id known → call lookup_order(order_id) immediately
- If order_id missing → ask for it
- Report result. If not found say "Order not found."

=== FLOW 2: ORDER HISTORY ===
- If customer_id known → call get_order_history(customer_id) immediately
- If customer_id missing → ask for it
- Show ALL orders returned

=== FLOW 3: REORDER ===
- Get order history first
- If 1 order: show it, ask "Would you like to reorder this?"
- If multiple: list all, ask "Which order would you like to reorder?"
- Wait for customer to pick
- Ask quantity if not given
- Wait for confirm
- Only then call create_reorder(order_id, quantity)

=== FLOW 4: REFUND ===
Step 1: call check_refund_policy(order_id)
- If NOT_ELIGIBLE → tell reason, ask if they want human support. STOP.
- If ELIGIBLE → go to Step 2

Step 2: call get_order_amount(order_id)
- If amount > 500 → call escalate_to_human("Refund amount $X exceeds $500 for order Y"). STOP.
- If amount <= 500 → ask customer "Would you like me to process this refund?"

Step 3: Only after customer says yes → call process_refund(order_id)
- Confirm success to customer

=== FLOW 5: HUMAN SUPPORT ===
- If customer asks for human → call escalate_to_human("Customer requested human support") immediately

Respond ONLY with raw JSON:

Call tool:
{"action": "tool", "message": "<brief status>", "tool_calls": [{"tool": "<name>", "arguments": {}}]}

Reply:
{"action": "reply", "message": "<reply>", "tool_calls": []}

=== EXAMPLES ===

User: "where is my order [Known IDs: {"order_id": "1234"}]"
{"action": "tool", "message": "Looking up order 1234.", "tool_calls": [{"tool": "lookup_order", "arguments": {"order_id": "1234"}}]}

Tool: "Order ID: 1234, Item: Wireless Headphones, Status: Shipped - Arriving Tomorrow, Amount: $89.99"
{"action": "reply", "message": "Your order 1234 (Wireless Headphones, $89.99) has shipped and will arrive tomorrow.", "tool_calls": []}

User: "show my orders [Known IDs: {"customer_id": "usr_99"}]"
{"action": "tool", "message": "Getting order history.", "tool_calls": [{"tool": "get_order_history", "arguments": {"customer_id": "usr_99"}}]}

Tool: "Order ID: 1234, Item: Wireless Headphones, Status: Shipped, Amount: $89.99\nOrder ID: 5678, Item: Gaming Mouse, Status: Delivered, Amount: $49.99"
{"action": "reply", "message": "Here are your orders:\n1. Order 1234 - Wireless Headphones ($89.99, Shipped)\n2. Order 5678 - Gaming Mouse ($49.99, Delivered)", "tool_calls": []}

User: "refund my order [Known IDs: {"order_id": "1234"}]"
{"action": "tool", "message": "Checking refund eligibility.", "tool_calls": [{"tool": "check_refund_policy", "arguments": {"order_id": "1234"}}]}

Tool: "ELIGIBLE"
{"action": "tool", "message": "Checking order amount.", "tool_calls": [{"tool": "get_order_amount", "arguments": {"order_id": "1234"}}]}

Tool: "89.99"
{"action": "reply", "message": "Your order 1234 is eligible for a refund of $89.99. Would you like me to go ahead and process it?", "tool_calls": []}

User: "yes [Known IDs: {"order_id": "1234"}]"
{"action": "tool", "message": "Processing your refund.", "tool_calls": [{"tool": "process_refund", "arguments": {"order_id": "1234"}}]}

Tool: "Refund successfully processed for order 1234"
{"action": "reply", "message": "Your refund of $89.99 for order 1234 has been successfully processed!", "tool_calls": []}

User: "refund my order [Known IDs: {"order_id": "9941"}]"
{"action": "tool", "message": "Checking refund eligibility.", "tool_calls": [{"tool": "check_refund_policy", "arguments": {"order_id": "9941"}}]}

Tool: "NOT_ELIGIBLE: Purchase older than 30 days"
{"action": "reply", "message": "Sorry, order 9941 is not eligible for a refund because it was purchased more than 30 days ago. Would you like me to escalate this to a human agent?", "tool_calls": []}

User: "I want to reorder [Known IDs: {"customer_id": "usr_99"}]"
{"action": "tool", "message": "Getting your order history.", "tool_calls": [{"tool": "get_order_history", "arguments": {"customer_id": "usr_99"}}]}

Tool: "Order ID: 1234...\\nOrder ID: 5678..."
{"action": "reply", "message": "Here are your previous orders:\n1. Order 1234 - Wireless Headphones ($89.99)\n2. Order 5678 - Gaming Mouse ($49.99)\nWhich order would you like to reorder?", "tool_calls": []}
"""

# ── Globals ────────────────────────────────────────────────────────────────────
mcp_session: ClientSession | None = None
coordinator: Coordinator | None = None
_stdio_ctx = None
_session_ctx = None


def _fresh_single_state():
    return {
        "messages": [{"role": "system", "content": SINGLE_AGENT_PROMPT}],
        "known_ids": {},
    }


single_state = _fresh_single_state()
multi_state = {"shared_state": {}, "conversation_log": ""}

OFFTOPIC_REPLY = (
    "I can only help with order and retail related questions. "
    "How can I assist you with your orders or account?"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_session, coordinator, _stdio_ctx, _session_ctx

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(os.path.dirname(__file__), "server.py")],
        env=os.environ.copy(),
    )
    _stdio_ctx = stdio_client(server_params)
    read_stream, write_stream = await _stdio_ctx.__aenter__()
    _session_ctx = ClientSession(read_stream, write_stream)
    mcp_session = await _session_ctx.__aenter__()
    await mcp_session.initialize()

    agents = {
        "orders_agent":  OrdersAgent(mcp_session),
        "refunds_agent": RefundsAgent(mcp_session),
        "account_agent": AccountAgent(mcp_session),
        "support_agent": SupportAgent(mcp_session),
    }
    coordinator = Coordinator(agents)
    print("✅ Both agent systems ready.")
    yield

    await _session_ctx.__aexit__(None, None, None)
    await _stdio_ctx.__aexit__(None, None, None)


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str

class ToolCallOut(BaseModel):
    tool: str
    arguments: dict
    result: str

class AgentActivity(BaseModel):
    agent_name: str
    status: str
    summary: str
    tool_calls: list[ToolCallOut]

class ChatResponse(BaseModel):
    reply: str
    agent_activity: list[AgentActivity]
    shared_state: dict


def build_agent_activity(agent_results: list) -> list[AgentActivity]:
    return [
        AgentActivity(
            agent_name=r.agent_name,
            status=r.status,
            summary=r.summary,
            tool_calls=[
                ToolCallOut(tool=t["tool"], arguments=t["arguments"], result=t["result"])
                for t in r.tool_log
            ],
        )
        for r in agent_results
    ]


def sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


async def execute_tool_single(tool_name: str, arguments: dict) -> str:
    try:
        result = await mcp_session.call_tool(tool_name, arguments)
        return result.content[0].text
    except Exception as exc:
        return f"ERROR calling {tool_name}: {exc}"


@app.post("/chat/single", response_model=ChatResponse)
async def chat_single(request: ChatRequest):
    if mcp_session is None:
        raise HTTPException(status_code=503, detail="Not ready")

    # 1. Python intent check
    intent = _detect_intent(request.message)
    if intent == "offtopic":
        return ChatResponse(reply=OFFTOPIC_REPLY, agent_activity=[], shared_state={})

    # 2. Extract IDs in Python
    id_updates = _extract_ids(request.message, single_state["known_ids"])
    single_state["known_ids"].update(id_updates)

    # 3. Inject known IDs into message
    known = single_state["known_ids"]
    user_content = request.message
    if known:
        user_content += f"\n[Known IDs: {json.dumps(known)}]"

    messages = single_state["messages"]
    tool_log_this_turn = []
    messages.append({"role": "user", "content": user_content})

    for _ in range(6):
        response = ollama_chat(model=MODEL_NAME, messages=messages, format="json")
        content = response["message"]["content"]
        parsed = safe_json_loads(content)

        if parsed is None:
            messages.append({"role": "assistant", "content": content})
            return ChatResponse(
                reply="Sorry, I had trouble with that. Could you rephrase?",
                agent_activity=[AgentActivity(
                    agent_name="single_agent", status="error",
                    summary="Unparseable output", tool_calls=[]
                )],
                shared_state=known,
            )

        action = parsed.get("action")
        message_text = parsed.get("message", "")
        tool_calls = parsed.get("tool_calls", []) or []

        if action == "reply" or not tool_calls:
            messages.append({"role": "assistant", "content": json.dumps(parsed)})
            return ChatResponse(
                reply=message_text,
                agent_activity=[AgentActivity(
                    agent_name="single_agent", status="done",
                    summary=message_text,
                    tool_calls=[ToolCallOut(**t) for t in tool_log_this_turn],
                )],
                shared_state=known,
            )

        messages.append({"role": "assistant", "content": json.dumps(parsed)})

        tool_results = []
        for call in tool_calls:
            tool_name = call.get("tool")
            arguments = call.get("arguments", {}) or {}

            # Fallback: inject known IDs if model forgot to use them
            if "order_id" not in arguments and "order_id" in known:
                if any(x in tool_name for x in ["order", "refund", "amount"]):
                    arguments["order_id"] = known["order_id"]
            if "customer_id" not in arguments and "customer_id" in known:
                if any(x in tool_name for x in ["history", "customer", "reorder"]):
                    arguments["customer_id"] = known["customer_id"]

            result = await execute_tool_single(tool_name, arguments)
            tool_log_this_turn.append({"tool": tool_name, "arguments": arguments, "result": result})
            tool_results.append({"tool": tool_name, "result": result})

        messages.append({
            "role": "user",
            "content": (
                f"Tool results:\n{json.dumps(tool_results, indent=2)}\n\n"
                f"Known IDs: {json.dumps(known)}\n"
                "Respond with JSON. Use action=reply if you have the answer."
            ),
        })

    return ChatResponse(
        reply="Sorry, I'm having trouble completing that. Please try again.",
        agent_activity=[],
        shared_state=known,
    )


@app.post("/chat/multi", response_model=ChatResponse)
async def chat_multi(request: ChatRequest):
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Not ready")

    reply, multi_state["shared_state"], agent_results = await coordinator.handle_turn(
        multi_state["conversation_log"],
        multi_state["shared_state"],
        request.message,
    )
    multi_state["conversation_log"] += f"\nCustomer: {request.message}\nCoordinator: {reply}"

    activity = build_agent_activity(agent_results)

    return ChatResponse(
        reply=reply,
        agent_activity=activity,
        shared_state=multi_state["shared_state"],
    )


@app.post("/chat/multi/stream")
async def chat_multi_stream(request: ChatRequest):
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Not ready")

    async def event_generator():
        try:
            turn = await coordinator.prepare_turn(
                multi_state["conversation_log"],
                multi_state["shared_state"],
                request.message,
            )
            activity = build_agent_activity(turn.agent_results)
            yield sse_event("activity", {
                "agent_activity": [model_to_dict(item) for item in activity],
                "shared_state": turn.shared_state,
            })

            chunks = []
            for chunk in coordinator.synthesize_stream(
                request.message,
                turn.agent_results,
                turn.reply_context,
            ):
                chunks.append(chunk)
                yield sse_event("token", {"text": chunk})

            reply = "".join(chunks).strip()
            multi_state["shared_state"] = turn.shared_state
            multi_state["conversation_log"] += f"\nCustomer: {request.message}\nCoordinator: {reply}"

            yield sse_event("done", {
                "reply": reply,
                "agent_activity": [model_to_dict(item) for item in activity],
                "shared_state": multi_state["shared_state"],
            })
        except Exception as exc:
            yield sse_event("error", {"message": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/reset")
async def reset():
    global single_state
    single_state = _fresh_single_state()
    multi_state["shared_state"] = {}
    multi_state["conversation_log"] = ""
    return {"status": "reset"}


@app.get("/health")
async def health():
    return {"status": "ok", "ready": coordinator is not None}
