"""System Promptfor tau2-bench."""

MULTI_TURN_INSTRUCTION = """You are a customer service agent that helps the user according to the <policy> provided below.

CRITICAL: You MUST use the `send_message` tool for ALL communication with the user. Never respond with plain text.

CRITICAL: You ARE the technical support agent. The <policy> contains YOUR procedures and tools. You have all the capabilities needed to resolve technical issues. Only transfer to a human agent if you have exhausted ALL troubleshooting steps in the policy and the issue still cannot be resolved.

In each turn you can either:
- Use the `send_message` tool to communicate with the user
- Make another tool call to access information or perform actions

You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""

def _format_system_prompt(policy: str, solo_mode: bool = False) -> str:
    """
    Format the TAU2-bench system prompt.

    Returns the agent system prompt in TAU2's format:
    - Multi-turn mode: Instructions + Policy (agent discovers task through conversation)
    - Solo mode: Instructions + Policy + Ticket (agent gets task description upfront)
    """
    if solo_mode:
        # Solo mode: includes ticket with task description
        agent_instruction = """You are a customer service agent that helps the user according to the <policy> provided below.
You will be provided with a ticket that contains the user's request.
You will need to plan and call the appropriate tools to solve the ticket.

You cannot communicate with the user, only make tool calls.
Stop when you consider that you have solved the ticket by calling the evaluate/evaluate_task tool.

Always follow the policy. Always make sure you generate valid JSON only."""

        # Note: In solo mode, ticket would be added to system prompt
        return f"""<instructions>
{agent_instruction}
</instructions>

<policy>
{policy}
</policy>

Note: The task/ticket description will be provided in the initial prompt."""

    else:
        # Multi-turn mode: NO ticket, agent learns through conversation
        return f"""<instructions>
{MULTI_TURN_INSTRUCTION}
</instructions>

<policy>
{policy}
</policy>"""