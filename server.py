from fastmcp import FastMCP

mcp = FastMCP("EnterpriseRetailServer")

from mock_data import MOCK_CUSTOMERS, MOCK_ORDERS


def _norm(value) -> str:
    """
    Coerce incoming ids to a clean string.
    Small LLMs sometimes emit 1234 (int) instead of "1234" (str),
    or add stray whitespace. Dict lookups are exact-match, so a type
    mismatch silently looks identical to "not found". Normalize first.
    """
    return str(value).strip()


@mcp.tool()
def get_customer(customer_id: str) -> str:
    """
    Look up a customer by their customer_id.

    Args:
        customer_id: The customer's id, e.g. "usr_99". Always pass as a string,
            even if it looks numeric.

    Returns:
        A line of customer details, or "Customer not found" if no match exists.

    Example:
        get_customer(customer_id="usr_99")
        -> "Customer ID: usr_99, Name: John Doe, Past Refunds: 0"
    """
    customer_id = _norm(customer_id)
    customer = MOCK_CUSTOMERS.get(customer_id)

    if not customer:
        return f"Customer not found for customer_id={customer_id}"

    return (
        f"Customer ID: {customer_id}, "
        f"Name: {customer['name']}, "
        f"Past Refunds: {customer['past_refunds']}"
    )


@mcp.tool()
def lookup_order(order_id: str) -> str:
    """
    Look up an order's status and details by order_id.

    Args:
        order_id: The order's id, e.g. "ORD1234". Always pass as a string,
            even if it looks numeric (do NOT pass ORD1234, pass "ORD1234").

    Returns:
        A line with item, status, and amount, or "Order not found" if no match.

    Example:
        lookup_order(order_id="ORD1234")
        -> "Order ID: ORD1234, Item: Wireless Headphones, Status: Shipped - Arriving Tomorrow, Amount: $89.99"
    """
    order_id = _norm(order_id)
    order = MOCK_ORDERS.get(order_id)

    if not order:
        return f"Order not found for order_id={order_id}"

    return (
        f"Order ID: {order_id}, "
        f"Item: {order['item']}, "
        f"Status: {order['status']}, "
        f"Amount: ${order['amount']}"
    )


@mcp.tool()
def check_refund_policy(order_id: str) -> str:
    """
    Check whether an order is eligible for a refund.
    Always call this BEFORE process_refund.

    Args:
        order_id: The order's id, e.g. "ORD1234". Always pass as a string.

    Returns:
        "ELIGIBLE" if the order qualifies for a refund,
        "NOT_ELIGIBLE: <reason>" if it does not (already refunded, or
        purchase too old),
        or "Order not found" if no match.

    Example:
        check_refund_policy(order_id="ORD1234")
        -> "ELIGIBLE"

        check_refund_policy(order_id="ORD9941")
        -> "NOT_ELIGIBLE: Purchase older than 30 days"
    """
    order_id = _norm(order_id)
    order = MOCK_ORDERS.get(order_id)

    if not order:
        return f"Order not found for order_id={order_id}"

    if order["refunded"]:
        return "NOT_ELIGIBLE: This order has already been refunded"

    if order["days_since_purchase"] > 30:
        return "NOT_ELIGIBLE: Purchase older than 30 days"

    return "ELIGIBLE"


@mcp.tool()
def process_refund(order_id: str) -> str:
    """
    Process a refund for an order. Only call this AFTER
    check_refund_policy has returned "ELIGIBLE" for the same order_id.

    Args:
        order_id: The order's id, e.g. "ORD1234". Always pass as a string.

    Returns:
        A confirmation message, or an error message if the order
        doesn't exist or was already refunded.

    Example:
        process_refund(order_id="ORD1234")
        -> "Refund successfully processed for order ORD1234"
    """
    order_id = _norm(order_id)
    order = MOCK_ORDERS.get(order_id)

    if not order:
        return f"Order not found for order_id={order_id}"

    if order["refunded"]:
        return f"Order {order_id} was already refunded earlier; cannot refund again"

    order["refunded"] = True

    return f"Refund successfully processed for order {order_id}"


@mcp.tool()
def escalate_to_human(reason: str) -> str:
    """
    Escalate the current issue to a human support agent.
    Use this when a refund is not eligible, or when the customer
    explicitly asks to speak to a human.

    Args:
        reason: A short plain-text reason for the escalation.

    Returns:
        A confirmation message.

    Example:
        escalate_to_human(reason="Refund not eligible: purchase older than 30 days")
        -> "Escalated to human support. Reason: Refund not eligible: purchase older than 30 days"
    """
    return f"Escalated to human support. Reason: {reason}"


@mcp.tool()
def get_order_amount(order_id: str) -> str:
    """
    Get the order amount for a specific order.
    Args:
        order_id: The order ID.
    Returns:
        Order amount as string.
    Example:
        get_order_amount(order_id="ORD1234")
        -> "89.99"
    """
    order_id = _norm(order_id)

    order = MOCK_ORDERS.get(order_id)

    if not order:
        return f"Order not found for order_id={order_id}"

    return str(order["amount"])


@mcp.tool()
def get_order_history(customer_id: str) -> str:
    """
    Get all orders belonging to a customer.
    Args:
        customer_id: Customer ID.
    Returns:
        List of orders for that customer.
    Example:
        get_order_history(customer_id="usr_99")
    """
    customer_id = _norm(customer_id)

    orders = []

    for order_id, order in MOCK_ORDERS.items():
        if order["customer_id"] == customer_id:
            orders.append(
                f"Order ID: {order_id}, "
                f"Item: {order['item']}, "
                f"Status: {order['status']}, "
                f"Amount: ${order['amount']}"
            )

    if not orders:
        return f"No orders found for customer {customer_id}"

    return "\n".join(orders)


@mcp.tool()
def create_reorder(order_id: str, quantity: int = 1) -> str:
    """
    Create a reorder from a previous order.
    Args:
        order_id: Previous order ID.
        quantity: Quantity to reorder.
    Returns:
        Reorder confirmation.
    Example:
        create_reorder(
            order_id="ORD5678",
            quantity=6
        )
    """
    order_id = _norm(order_id)

    order = MOCK_ORDERS.get(order_id)

    if not order:
        return f"Order not found for order_id={order_id}"

    return (
        f"Reorder created successfully. "
        f"Item: {order['item']}, "
        f"Quantity: {quantity}, "
        f"Total Amount: ${order['amount'] * quantity:.2f}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
