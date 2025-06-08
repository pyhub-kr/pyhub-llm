import typer
from rich.console import Console

from pyhub.llm.commands.agent import app as agent_app
from pyhub.llm.commands.ask import ask
from pyhub.llm.commands.chat import chat
from pyhub.llm.commands.compare import compare
from pyhub.llm.commands.describe import describe
from pyhub.llm.commands.embed import app as embed_app

# from pyhub import print_for_main


app = typer.Typer()
console = Console()

app.add_typer(embed_app, name="embed")
app.add_typer(agent_app, name="agent")

app.command()(ask)
app.command()(describe)
app.command()(chat)
app.command()(compare)


logo = """
    ██████╗  ██╗   ██╗ ██╗  ██╗ ██╗   ██╗ ██████╗     ██╗      ██╗      ███╗   ███╗
    ██╔══██╗ ╚██╗ ██╔╝ ██║  ██║ ██║   ██║ ██╔══██╗    ██║      ██║      ████╗ ████║
    ██████╔╝  ╚████╔╝  ███████║ ██║   ██║ ██████╔╝    ██║      ██║      ██╔████╔██║
    ██╔═══╝    ╚██╔╝   ██╔══██║ ██║   ██║ ██╔══██╗    ██║      ██║      ██║╚██╔╝██║
    ██║         ██║    ██║  ██║ ╚██████╔╝ ██████╔╝    ███████╗ ███████╗ ██║ ╚═╝ ██║
    ╚═╝         ╚═╝    ╚═╝  ╚═╝  ╚═════╝  ╚═════╝     ╚══════╝ ╚══════╝ ╚═╝     ╚═╝
"""

# app.callback(invoke_without_command=True)(print_for_main(logo))
