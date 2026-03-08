from __future__ import annotations

import json
import re

from schemas import ParsedAssistantOutput


class OutputParser:
    """Parses and validates structured assistant JSON outputs."""

    def repair_json(self, raw: str) -> str:
        text = raw.strip()
        if not text.startswith("{"):
            start = text.find("{")
            if start >= 0:
                text = text[start:]
        text = re.sub(r",\s*}\s*$", "}", text)
        text = re.sub(r",\s*]\s*$", "]", text)
        return text

    def parse(self, raw: str) -> ParsedAssistantOutput:
        repaired = self.repair_json(raw)
        data = json.loads(repaired)
        return ParsedAssistantOutput.model_validate(data)
