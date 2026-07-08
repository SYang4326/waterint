from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class OutputPrinter(Protocol):
    def __call__(self, result: Any) -> None:
        ...


@dataclass(frozen=True)
class AnalysisModule:
    """User-facing analysis method exposed through the CLI, UI, or docs."""

    name: str
    help: str
    run: Callable[[dict[str, Any]], Any]
    print_outputs: OutputPrinter
    aliases: tuple[str, ...] = ()
    config_help: str = "YAML config file."
    description: str = ""

    @property
    def command_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    def add_cli_parser(self, subparsers: Any) -> None:
        for command in self.command_names:
            help_text = self.help
            if command != self.name:
                help_text = f"Alias for {self.name}. {self.help}"
            parser = subparsers.add_parser(command, help=help_text, description=self.description or None)
            parser.add_argument("--config", required=True, help=self.config_help)
            parser.set_defaults(analysis_module=self)

    def run_config(self, config: dict[str, Any]) -> Any:
        return self.run(config)
