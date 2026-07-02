import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.orders_agent import OrdersAgent
from agents.refunds_agent import RefundsAgent
from agents.account_agent import AccountAgent
from agents.support_agent import SupportAgent
from coordinator import Coordinator


def format_agent_log(agent_results):
    """Pretty debug print of what each agent did this turn."""
    if not agent_results:
        return
    print("\n--- agent activity this turn ---")
    for r in agent_results:
        print(f"[{r.agent_name}] status={r.status}")
        if r.summary:
            print(f"    summary: {r.summary}")
        for t in r.tool_log:
            print(f"    tool: {t['tool']}({t['arguments']}) -> {t['result']}")
    print("--------------------------------")


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            agents = {
                "orders_agent": OrdersAgent(session),
                "refunds_agent": RefundsAgent(session),
                "account_agent": AccountAgent(session),
                "support_agent": SupportAgent(session),
            }
            coordinator = Coordinator(agents)

            shared_state = {}     # e.g. {"order_id": "1234", "customer_id": "usr_99"}
            conversation_log = "" # plain-text rolling transcript for routing context

            print("\n=========================================")
            print(" Multi-Agent Customer Service System")
            print(" (coordinator + orders/refunds/account/support agents)")
            print("=========================================")

            while True:
                user_input = input("\nCustomer > ")
                if user_input.lower() in ["exit", "quit"]:
                    break

                reply, shared_state, agent_results = await coordinator.handle_turn(
                    conversation_log, shared_state, user_input
                )

                format_agent_log(agent_results)
                print(f"\nAgent: {reply}")

                conversation_log += f"\nCustomer: {user_input}\nCoordinator: {reply}"


if __name__ == "__main__":
    asyncio.run(main())
