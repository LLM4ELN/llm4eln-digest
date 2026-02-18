"""Tests for llm4eln_digest.tools.basic_tools."""

from __future__ import annotations

import pytest

from llm4eln_digest.tools.basic_tools import (
    AVAILABLE_TOOLS,
    CalculateMetadataStatsTool,
    GetCurrentTimeTool,
    UpdatePreviewTool,
)

# ---------------------------------------------------------------------------
# GetCurrentTimeTool
# ---------------------------------------------------------------------------


class TestGetCurrentTimeTool:
    def setup_method(self) -> None:
        self.tool = GetCurrentTimeTool()

    def test_run_contains_current_time(self) -> None:
        result = self.tool._run()
        assert "Current time" in result

    def test_run_includes_timezone(self) -> None:
        result = self.tool._run(timezone="US/Eastern")
        assert "US/Eastern" in result

    @pytest.mark.asyncio()
    async def test_arun_matches_run(self) -> None:
        async_result = await self.tool._arun(timezone="UTC")
        # Exact second may differ between calls, so just check the prefix
        assert async_result.startswith("Current time (UTC):")

    def test_default_timezone_is_utc(self) -> None:
        result = self.tool._run()
        assert "UTC" in result


# ---------------------------------------------------------------------------
# CalculateMetadataStatsTool
# ---------------------------------------------------------------------------


class TestCalculateMetadataStatsTool:
    def setup_method(self) -> None:
        self.tool = CalculateMetadataStatsTool()

    def test_zero_entries(self) -> None:
        result = self.tool._run(entry_count=0, has_metadata=0)
        assert result["total_entries"] == 0
        assert result["coverage_percentage"] == 0.0
        assert result["missing_metadata"] == 0

    def test_normal_case(self) -> None:
        result = self.tool._run(entry_count=10, has_metadata=7)
        assert result["total_entries"] == 10
        assert result["entries_with_metadata"] == 7
        assert result["coverage_percentage"] == 70.0
        assert result["missing_metadata"] == 3

    def test_full_coverage(self) -> None:
        result = self.tool._run(entry_count=5, has_metadata=5)
        assert result["coverage_percentage"] == 100.0
        assert result["missing_metadata"] == 0

    @pytest.mark.asyncio()
    async def test_arun_matches_run(self) -> None:
        sync_result = self.tool._run(entry_count=4, has_metadata=2)
        async_result = await self.tool._arun(entry_count=4, has_metadata=2)
        assert sync_result == async_result


# ---------------------------------------------------------------------------
# UpdatePreviewTool
# ---------------------------------------------------------------------------


class TestUpdatePreviewTool:
    def setup_method(self) -> None:
        self.tool = UpdatePreviewTool()

    def test_returns_marker_format(self) -> None:
        result = self.tool._run(content="hello")
        assert result.startswith("PREVIEW_UPDATE::")

    def test_default_title(self) -> None:
        result = self.tool._run(content="hello")
        assert result == "PREVIEW_UPDATE::Preview::hello"

    def test_custom_title(self) -> None:
        result = self.tool._run(content="body", title="My Title")
        assert result == "PREVIEW_UPDATE::My Title::body"

    @pytest.mark.asyncio()
    async def test_arun_matches_run(self) -> None:
        sync_result = self.tool._run(content="data", title="T")
        async_result = await self.tool._arun(content="data", title="T")
        assert sync_result == async_result


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_available_tools_count(self) -> None:
        assert len(AVAILABLE_TOOLS) == 3

    def test_tool_types(self) -> None:
        types = {type(t) for t in AVAILABLE_TOOLS}
        assert types == {GetCurrentTimeTool, CalculateMetadataStatsTool, UpdatePreviewTool}
