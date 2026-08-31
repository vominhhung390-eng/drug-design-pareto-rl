import re
import pytest

from reinvent.scoring.utils import suppress_output, camel_to_snake


def test_suppress_output_does_not_raise():
    with suppress_output():
        print("stdout suppressed")
        import sys

        print("stderr suppressed", file=sys.stderr)


def test_suppress_output_yields_streams():
    with suppress_output() as (err, out):
        assert err is not None
        assert out is not None


@pytest.mark.parametrize(
    "input_name, expected",
    [
        ("CamelCase", "camel_case"),
        ("HTMLParser", "html_parser"),
        ("alreadysnake", "alreadysnake"),
        ("MyModelV2", "my_model_v2"),
        ("MolecularWeight", "molecular_weight"),
        ("QED", "qed"),
        ("SlogP", "slog_p"),
    ],
)
def test_camel_to_snake(input_name, expected):
    assert camel_to_snake(input_name) == expected
