"""Command-line interface for startup idea generation."""

import os
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .idea_generator.generator import StartupIdeaGenerator
from .utils.caching import clear_cache

app = typer.Typer(
    name="rag_startups",
    help="Generate and analyze startup ideas using AI and market data.",
    add_completion=False,
)
console = Console()


def validate_token() -> str:
    """Validate HuggingFace token is available."""
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        console.print(
            "[red]Error:[/red] HUGGINGFACE_TOKEN environment variable not set.\n"
            "Please set it with your HuggingFace API token."
        )
        raise typer.Exit(1)
    return token


def display_idea(idea_text: str, market_insights: Optional[dict] = None):
    """Display generated idea with optional market insights."""
    # Display the idea
    console.print(Panel(idea_text, title="Generated Startup Idea", expand=False))

    # Display market insights if available
    if market_insights:
        table = Table(
            title="Market Analysis", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for name, insights in market_insights.items():
            table.add_row("Startup Name", name)
            table.add_row("Market Size", f"${insights.market_size:,.2f}")
            table.add_row("Growth Rate", f"{insights.growth_rate:.1f}%")
            table.add_row("Competition", insights.competition_level)
            table.add_row("Opportunity Score", f"{insights.opportunity_score:.2f}")
            table.add_row("Confidence Score", f"{insights.confidence_score:.2f}")

            if insights.key_trends:
                table.add_row(
                    "Key Trends",
                    "\n".join(f"• {trend}" for trend in insights.key_trends),
                )

            if insights.risk_factors:
                table.add_row(
                    "Risk Factors",
                    "\n".join(f"• {risk}" for risk in insights.risk_factors),
                )

        console.print(table)


@app.command()
def generate(
    num_ideas: int = typer.Option(
        1, "--num", "-n", help="Number of startup ideas to generate (1-5)"
    ),
    market_analysis: bool = typer.Option(
        True, "--market/--no-market", help="Include market analysis"
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Model temperature (0.0-1.0). Higher values make output more creative.",
    ),
):
    """Generate startup ideas with optional market analysis."""
    # Validate number of ideas
    if not 1 <= num_ideas <= 5:
        console.print("[red]Error:[/red] num_ideas must be between 1 and 5")
        raise typer.Exit(1)

    # Validate token
    token = validate_token()

    with console.status("[bold green]Generating startup ideas..."):
        generator = StartupIdeaGenerator(token=token)
        response, insights = generator.generate(
            num_ideas=num_ideas,
            temperature=temperature,
            include_market_analysis=market_analysis,
        )

    if response:
        display_idea(response, insights if market_analysis else None)
    else:
        console.print("[red]Failed to generate startup ideas.[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    idea_name: str = typer.Argument(..., help="Name of the startup to analyze"),
    idea_description: str = typer.Argument(
        ..., help="Brief description of the startup idea"
    ),
    target_market: str = typer.Argument(..., help="Target market for the startup"),
):
    """Analyze market potential for an existing startup idea."""
    token = validate_token()

    with console.status("[bold green]Analyzing market potential..."):
        generator = StartupIdeaGenerator(token=token)
        idea = {
            "name": idea_name,
            "description": idea_description,
            "target_market": target_market,
        }
        _, insights = generator.generate(
            num_ideas=1,
            example_startups=[idea],
            include_market_analysis=True,
        )

    if insights:
        display_idea(
            f"Analysis for: {idea_name}\n\n{idea_description}\nTarget Market: {target_market}",
            insights,
        )
    else:
        console.print("[red]Failed to analyze startup idea.[/red]")
        raise typer.Exit(1)


@app.command()
def clear():
    """Clear cached data and analysis results."""
    try:
        clear_cache()
        console.print("[green]Successfully cleared cache.[/green]")
    except Exception as e:
        console.print(f"[red]Failed to clear cache: {e}[/red]")
        raise typer.Exit(1)


def main():
    """Entry point for the CLI."""
    app()
