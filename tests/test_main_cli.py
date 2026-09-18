from __future__ import annotations

from layercake.__main__ import main


def test_root_help_is_a_successful_discovery_surface(capsys) -> None:
    assert main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "LayerCake modular language-model host" in captured.out
    assert "cake" in captured.out
    assert "run" in captured.out
    assert captured.err == ""


def test_no_command_prints_help_and_fails(capsys) -> None:
    assert main([]) == 2
    captured = capsys.readouterr()
    assert "usage: layercake" in captured.err


def test_unknown_command_prints_actionable_help(capsys) -> None:
    assert main(["not-a-command"]) == 2
    captured = capsys.readouterr()
    assert "unknown command: not-a-command" in captured.err
    assert "layercake cake --help" in captured.err
