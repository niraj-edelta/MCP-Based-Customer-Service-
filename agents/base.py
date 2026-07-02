import json
from dataclasses import dataclass, field

from ollama import chat

from agents.json_utils import safe_json_loads


MODEL_NAME = "llama3.2:3b"


@dataclass
class AgentResult:
    """
    What a sub-agent hands back to the coordinator. Deliberately structured
    (not free text) so the coordinator can merge several of these into one
    customer-facing reply without re-parsing prose.
    """
    agent_name: str
    status: str            # "done" | "needs_input" | "error"
    summary: str            # short factual summary of what happened, grounded in tool results
    needs_input_prompt: str = ""   # filled in only if status == "needs_input"
    tool_log: list = field(default_factory=list)  # raw tool calls/results, for debugging/audit
    state_updates: dict = field(default_factory=dict)  # e.g. {"order_id": "1234"}


class BaseAgent:
    """
    A domain-scoped agent: owns a fixed set of MCP tools, a system prompt
    describing its own job only, and a short bounded reasoning loop.

    Sub-agents NEVER decide which other agent to call -- that's the
    coordinator's job. A sub-agent either finishes its own slice of the
    task, or reports that it needs more info from the customer.
    """

    name: str = "base"
    allowed_tools: list = []
    system_prompt: str = ""
    max_loops: int = 4

    def __init__(self, mcp_session):
        self.session = mcp_session

    async def _call_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name not in self.allowed_tools:
            return (
                f"ERROR: {self.name} is not permitted to call tool "
                f"'{tool_name}'. Allowed tools: {self.allowed_tools}"
            )
        try:
            result = await self.session.call_tool(tool_name, arguments)
            return result.content[0].text
        except Exception as exc:
            return f"ERROR calling {tool_name}: {exc}"

    async def run(self, task_brief: str, shared_state: dict) -> AgentResult:
        """
        task_brief: a plain-English instruction from the coordinator describing
            exactly what this agent has been asked to do for this turn
            (e.g. "Check order status and refund eligibility for order 1234").
        shared_state: known facts carried across the whole conversation
            (e.g. {"order_id": "1234", "customer_id": "usr_99"}), so the
            agent doesn't re-ask for things already provided.
        """
        tool_log = []

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Known context so far: {json.dumps(shared_state)}\n\n"
                    f"Task from coordinator: {task_brief}\n\n"
                    "Respond ONLY with JSON as instructed in your system prompt."
                ),
            },
        ]

        for _ in range(self.max_loops):
            response = chat(model=MODEL_NAME, messages=messages, format="json")
            content = response["message"]["content"]
            parsed = safe_json_loads(content)

            if parsed is None:
                return AgentResult(
                    agent_name=self.name,
                    status="error",
                    summary=f"{self.name} returned unparseable output: {content[:200]}",
                    tool_log=tool_log,
                )

            action = parsed.get("action")
            tool_calls = parsed.get("tool_calls", []) or []

            if action == "needs_input":
                return AgentResult(
                    agent_name=self.name,
                    status="needs_input",
                    summary=parsed.get("summary", ""),
                    needs_input_prompt=parsed.get("prompt", "Could you provide more details?"),
                    tool_log=tool_log,
                )

            if action == "done" or not tool_calls:
                return AgentResult(
                    agent_name=self.name,
                    status="done",
                    summary=parsed.get("summary", ""),
                    tool_log=tool_log,
                    state_updates=parsed.get("state_updates", {}) or {},
                )

            # action == "tool": execute every requested call this round
            messages.append({"role": "assistant", "content": json.dumps(parsed)})

            results_this_round = []
            for call in tool_calls:
                tool_name = call.get("tool")
                arguments = call.get("arguments", {}) or {}
                result_text = await self._call_tool(tool_name, arguments)
                tool_log.append({"tool": tool_name, "arguments": arguments, "result": result_text})
                results_this_round.append({"tool": tool_name, "result": result_text})

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Tool results (ground truth -- use only this data, "
                        "never invent or contradict it):\n"
                        + json.dumps(results_this_round, indent=2)
                        + "\n\nBased on these results, respond with your next JSON action "
                        "(\"done\" with a summary, \"needs_input\" if you're missing "
                        "something, or another \"tool\" call if more lookups are needed)."
                    ),
                }
            )

        return AgentResult(
            agent_name=self.name,
            status="error",
            summary=f"{self.name} could not complete the task within its step budget.",
            tool_log=tool_log,
        )
