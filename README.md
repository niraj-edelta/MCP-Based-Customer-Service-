# Multi-Agent Customer Service System

A coordinator-led multi-agent rebuild of the single-agent retail support bot.
Same MCP tools, same local model (`llama3.2:3b` via Ollama), but the work is
split across specialist agents that the coordinator routes to and combines.

## Architecture

```
Customer message
      |
      v
 Coordinator (LLM routing call)
      |
      |-- decides: reply directly? OR dispatch to 1+ agents?
      |
      v
 ┌─────────────┬──────────────┬──────────────┬──────────────┐
 │ orders_agent│ refunds_agent│ account_agent│ support_agent│
 │ status,     │ eligibility, │ customer     │ escalation   │
 │ history,    │ process      │ profile      │              │
 │ reorder     │ refund       │ lookup       │              │
 └─────────────┴──────────────┴──────────────┴──────────────┘
      |  each agent runs its own bounded tool loop
      v  and returns a structured AgentResult
 Coordinator (LLM synthesis call)
      |
      v
 One combined, natural reply to the customer
```

### Why this split
Each agent owns a disjoint set of MCP tools — no tool is shared between two
agents, so there's never ambiguity about who's allowed to call what:

- **orders_agent** — `lookup_order`, `get_order_history`, `create_reorder`, `get_order_amount`
- **refunds_agent** — `check_refund_policy`, `process_refund`
- **account_agent** — `get_customer`
- **support_agent** — `escalate_to_human`

### Coordinator (`coordinator.py`)
Two LLM calls per customer turn:

1. **Routing** — reads the conversation log, known shared state (e.g.
   `order_id`, `customer_id` already established), and the new message.
   Decides whether to reply directly (small talk, no lookup needed) or
   dispatch to one or more agents with a precise task brief each.
2. **Synthesis** — after all dispatched agents return, combines their
   structured `AgentResult`s into a single natural customer-facing reply.
   This is what makes multi-concern messages ("what's my order status AND
   can I get a refund on it?") come back as one coherent answer instead of
   two bolted-together paragraphs.

### Sub-agents (`agents/`)
Each sub-agent (`BaseAgent` subclass) gets:
- A system prompt scoped to *only its own job* and *only its own tools*.
- A bounded reasoning loop (`agents/base.py`) that calls Ollama, executes
  any tool calls via the shared MCP session, and feeds results back until
  the agent reports `done` or `needs_input`.
- A hard whitelist check — an agent's tool-call dispatcher will reject any
  tool name not in its `allowed_tools`, even if the model hallucinates one.

Critical behavior preserved from the original single-agent system:
- **Never process a refund without explicit customer confirmation** —
  refunds_agent will not call `process_refund` unless its task brief says
  the customer has already confirmed it.
- **Never escalate without explicit customer confirmation** — same pattern
  in support_agent.
- **Never state a fact not grounded in a tool result** — every agent's
  system prompt repeats this rule for its own domain.

### Shared state
`main.py` keeps a small dict (e.g. `{"order_id": "1234", "customer_id":
"usr_99"}`) that persists across turns and is passed into both the
coordinator's routing call and every dispatched agent, so customers aren't
asked to repeat an ID they already gave. Agents report back
`state_updates` with any ids they've confirmed exist.

## Files

```
server.py              MCP server exposing all retail tools (unchanged from original)
main.py                 Entry point: wires MCP session + agents + coordinator into a chat loop
coordinator.py          LLM-based router + multi-agent response synthesizer
agents/
  base.py               BaseAgent: shared tool-loop runner, AgentResult dataclass
  json_utils.py          Robust JSON parsing for small-model output
  orders_agent.py        Order status / history / reorder
  refunds_agent.py        Refund eligibility / processing
  account_agent.py        Customer profile lookups
  support_agent.py        Human escalation
requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b      # if you haven't already
python main.py
```

Try a multi-concern message to see the coordinator combine two agents'
results into one reply, e.g.:

```
Customer > what's the status of order 9941 and can I also get a refund on it?
```

This should dispatch both `orders_agent` (status) and `refunds_agent`
(eligibility) in the same turn, and the coordinator will weave both into
one message — reporting the order is delivered AND that it isn't
refund-eligible (>30 days), then asking if you'd like to escalate.

## Notes / things you may want to tune
- `BaseAgent.max_loops` (default 4) caps how many tool-call rounds a single
  agent can take per dispatch — raise it if a domain needs deeper chains.
- The coordinator currently does sequential `await agent.run(...)` calls
  per dispatch. If you want agents to run concurrently when multiple are
  dispatched in one turn, swap that loop for `asyncio.gather(...)` — left
  sequential here since they share one MCP `ClientSession`, and concurrent
  tool calls over the same stdio session can race.
- All agents currently use `llama3.2:3b`; since each prompt is now smaller
  and domain-scoped, you may get away with an even smaller/faster model per
  agent if you want to optimize latency further.
