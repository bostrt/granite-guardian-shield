import logging

import click
from httpx import Client
from llama_stack_client import Agent, AgentEventLogger, LlamaStackClient
from llama_stack_client.types.shared import UserMessage
from prompt_toolkit import PromptSession
from rich.live import Live
from rich.panel import Panel

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)


def reverse_word(input: str) -> str:
    """
    Reverse a word.

    :param input: some input string
    """
    return input[::-1]


def multiply_by_two(input: int) -> int:
    """
    Multiply a number by 2.

    :param input: some int parameter
    """
    return input * 2


@click.command()
@click.option(
    "--endpoint",
    help="The Llama Stack endpoint",
    default="http://localhost:8321",
)
@click.option(
    "--model",
    default="granite3.3:2b"
)
def main(endpoint: str, model: str):
    ls_client = LlamaStackClient(
        base_url=endpoint,
        http_client=Client(verify=False),
    )
    ls_agent = Agent(
        client=ls_client,
        input_shields=["input_guardian"],
        # output_shields=["output_guardian"],
        # tools=[multiply_by_two, reverse_word],
        model=model,
        instructions="You are a helpful AI Assistant.",
        enable_session_persistence=True,
    )
    ls_session = ls_agent.create_session("my-session")

    prompt_session = PromptSession(message="(user) ")

    while True:
        user_input = prompt_session.prompt()

        response = ls_agent.create_turn(
            messages=[UserMessage(content=user_input, role="user")],
            session_id=ls_session,
            stream=True,
        )

        all_content = ""
        with Live(Panel(renderable=all_content)) as live:
            for chunk in AgentEventLogger().log(response):
                print(chunk)
                if not chunk.content:
                    continue
                elif chunk.content and chunk.color == "red" and chunk.role is None:
                    raise Exception("Error generating response")

                if chunk.role == "tool_execution":
                    print(Panel(chunk.content, title="Sources", border_style="blue"))
                    continue
                if chunk.role == "shield_call":
                    if chunk.content == "No Violation":  # This phrase is hardcoded in llama-stack-client-python
                        continue
                    else:
                        all_content += "Safety Violation Occurred\n"

                all_content += chunk.content
                live.update(Panel(all_content))


if __name__ == "__main__":
    main()
