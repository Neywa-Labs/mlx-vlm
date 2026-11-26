import importlib.util
import json
from pathlib import Path

import pytest

_PARSER_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "parsers" / "minimax_m3.py"
)
_spec = importlib.util.spec_from_file_location(
    "minimax_m3_parser_under_test", _PARSER_PATH
)
minimax_m3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(minimax_m3)

NS = "]<]minimax[>["
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "validate_policy",
            "parameters": {
                "type": "object",
                "properties": {"policy": {"type": "object"}},
                "required": ["policy"],
            },
        },
    }
]


def _call(key):
    return (
        f'{NS}<tool_call>{NS}<invoke name="validate_policy">{NS}<policy>'
        f"{NS}<Condition>{NS}<StringEquals>{NS}<{key}>true{NS}</{key}>{NS}</StringEquals>{NS}</Condition>"
        f"{NS}</policy>{NS}</invoke>{NS}</tool_call>"
    )


@pytest.mark.parametrize(
    "key",
    [
        "aws:SourceAccount",
        "aws:ResourceTag/team",
        "aws:RequestTag/${TagKey}",
        "application/json",
    ],
)
def test_object_keys_with_json_legal_punctuation_are_kept(key):
    parsed = minimax_m3.parse_tool_call(_call(key), TOOLS)

    assert parsed["name"] == "validate_policy"
    arguments = json.loads(parsed["arguments"])
    assert arguments == {"policy": {"Condition": {"StringEquals": {key: "true"}}}}
