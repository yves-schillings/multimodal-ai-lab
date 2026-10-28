"""Verify documentation coverage detects missing nested contracts and parameters.

These tests parse small source fixtures without importing or running them. They
exercise the coverage tool, not the AI models or the scientific measurements.
"""
from pathlib import Path

from scripts.check_engineering_documentation import check_python, write_index


def test_coverage_reports_nested_function_and_class_without_docstrings(tmp_path):
    """Detect omissions inside conditional/nested scopes, not just public methods.

    Args:
        tmp_path: pytest Path receiving a source fixture; no project file is edited.

    Returns:
        None. Assertions verify that both missing contracts are reported.
    """
    source = tmp_path / "sample.py"
    source.write_text('''"""Fixture module."""
class Worker:
    def run(self):
        """Perform the scripted fixture action."""
        if True:
            def nested(value):
                return value
        return nested
''', encoding="utf-8")
    result = check_python(source, "sample.py")
    missing = {e["name"] for e in result["errors"] if e["problem"] == "missing docstring"}
    assert missing == {"Worker", "Worker.run.nested"}
    assert result["functions"] == 2 and result["classes"] == 1


def test_coverage_reports_missing_parameter_but_exempts_self(tmp_path):
    """Require every formal argument name while exempting the implicit receiver.

    Args:
        tmp_path: pytest Path receiving an unexecuted Python source fixture.

    Returns:
        None. Assertions verify missing limit and kwargs are detected.
    """
    source = tmp_path / "sample.py"
    source.write_text('''"""Fixture module."""
def run(self, value, *, limit=3, **kwargs):
    """Return value unchanged for this fixture."""
    return value
''', encoding="utf-8")
    result = check_python(source, "sample.py")
    assert len(result["errors"]) == 1
    assert result["errors"][0]["parameters"] == ["limit", "kwargs"]


def test_coverage_passes_documented_code_and_index_links_current_lines(tmp_path):
    """Generate an index only from real documented declarations and source lines.

    Args:
        tmp_path: pytest Path used as an isolated repository with a docs directory.

    Returns:
        None. Assertions verify the generated signature, line link and callback count.
    """
    source = tmp_path / "sample.py"
    source.write_text('''"""Fixture module."""
def run(value):
    """Return value through an inline identity callback."""
    # The callback returns its input unchanged.
    return (lambda item: item)(value)
''', encoding="utf-8")
    result = check_python(source, "sample.py")
    assert result["errors"] == [] and result["anonymous_callback_lines"] == [5]
    (tmp_path / "docs/engineering").mkdir(parents=True)
    output = write_index(tmp_path, [result], "static/engineering.js")
    text = Path(output).read_text(encoding="utf-8")
    assert "run(value)" in text and "../../sample.py#L2" in text
    assert "Anonymous call-site callbacks: 1" in text
