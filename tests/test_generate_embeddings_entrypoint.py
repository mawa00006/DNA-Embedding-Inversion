"""The shared entrypoint must refuse the slugs that keep their own generator.

Inspected as source rather than imported: the module pulls in torch through
``src.fm_engine``, which not every env running the test suite has.
"""

import ast
from pathlib import Path

import pytest

from src.model_registry import REGISTRY

GENERATE_DIR = Path(__file__).resolve().parents[1] / "generate"


def _main_body():
    tree = ast.parse((GENERATE_DIR / "generate_embeddings.py").read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert len(functions) == 1
    return functions[0].body


def test_entrypoint_refuses_unsupported_slugs_before_any_work():
    body = _main_body()
    guard = [
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assert) and "engine_supported" in ast.dump(node.test)
    ]
    work = [
        i
        for i, node in enumerate(body)
        if any(name in ast.dump(node) for name in ("make_embed_chunk", "run_generation"))
    ]
    assert len(guard) == 1
    assert work and guard[0] < min(work)


@pytest.mark.parametrize(
    "slug", [s for s, spec in REGISTRY.items() if not spec.engine_supported]
)
def test_unsupported_slugs_still_have_the_generator_they_are_redirected_to(slug):
    assert (GENERATE_DIR / f"generate_{slug}_embeddings.py").is_file()
