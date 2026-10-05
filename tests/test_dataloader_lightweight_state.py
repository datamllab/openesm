import ast
from pathlib import Path


DATALOADER_PATH = Path(__file__).resolve().parents[1] / "esm" / "dataloader.py"


def _class_method(tree, class_name, method_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found")


def _string_constants(node):
    return {
        item.value
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


def test_training_iterator_uses_lightweight_state_without_doc_buffer_copy():
    tree = ast.parse(DATALOADER_PATH.read_text(encoding="utf-8"))

    full_state = _class_method(tree, "StatefulBestFitDataLoader", "state_dict")
    lightweight_state = _class_method(
        tree, "StatefulBestFitDataLoader", "lightweight_state_dict"
    )
    iterator = _class_method(tree, "StatefulBestFitDataLoader", "__iter__")

    assert "doc_buffer" in _string_constants(full_state)
    assert "doc_buffer" not in _string_constants(lightweight_state)

    calls = [
        node
        for node in ast.walk(iterator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    ]
    called_methods = {node.func.attr for node in calls}
    assert "lightweight_state_dict" in called_methods
    assert "state_dict" not in called_methods
