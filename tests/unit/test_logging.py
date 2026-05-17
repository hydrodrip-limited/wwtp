"""Unit tests for ``wwtp.logging_cfg``."""

from __future__ import annotations

import json
import logging

import pytest


class TestGetLogger:
    def test_returns_logger_instance(self) -> None:
        from wwtp.logging_cfg import get_logger

        logger = get_logger("test.basic")
        assert isinstance(logger, logging.Logger)

    def test_repeated_calls_return_same_logger(self) -> None:
        from wwtp.logging_cfg import get_logger

        l1 = get_logger("test.same")
        l2 = get_logger("test.same")
        assert l1 is l2

    def test_no_duplicate_handlers_on_reimport(self) -> None:
        from wwtp.logging_cfg import get_logger

        logger = get_logger("test.no_dupes")
        n_handlers = len(logger.handlers)
        # Call again — must not add a second handler
        get_logger("test.no_dupes")
        assert len(logger.handlers) == n_handlers

    def test_human_format_produces_pipe_separated_output(
        self, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        monkeypatch.setenv("LOG_FORMAT", "human")
        # Use a unique name so it gets a fresh handler
        from wwtp.logging_cfg import get_logger

        logger = get_logger("test.human_fmt")
        logger.info("hello human")
        captured = capsys.readouterr().out
        assert "|" in captured
        assert "hello human" in captured

    def test_json_format_produces_valid_json(self, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        monkeypatch.setenv("LOG_FORMAT", "json")
        # Fresh logger name to avoid collision with cached human-format handler
        from wwtp.logging_cfg import _JsonFormatter, get_logger

        logger = logging.getLogger("test.json_fmt_isolated")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(_JsonFormatter())
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

        logger.info("hello json")
        captured = capsys.readouterr().out
        # Each line must be valid JSON
        for line in captured.strip().splitlines():
            data = json.loads(line)
            assert "message" in data
            assert "level" in data


class TestJsonFormatter:
    def test_format_returns_valid_json(self) -> None:
        from wwtp.logging_cfg import _JsonFormatter

        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="test message %s",
            args=("arg1",),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "test message arg1"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"

    def test_format_includes_exc_info_when_present(self) -> None:
        from wwtp.logging_cfg import _JsonFormatter

        formatter = _JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="error occurred",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exc_info" in data
        assert "ValueError" in data["exc_info"]
