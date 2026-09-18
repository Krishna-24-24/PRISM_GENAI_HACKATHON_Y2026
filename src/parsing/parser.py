import hashlib
import re
from typing import Any, List, Set, Tuple

import tree_sitter
import tree_sitter_javascript

from src.chunking.models import CodeChunk

JS_LANGUAGE = tree_sitter.Language(tree_sitter_javascript.language())


def get_parser() -> tree_sitter.Parser:
    """Return a Tree-sitter parser initialized with the JavaScript grammar."""
    return tree_sitter.Parser(JS_LANGUAGE)


def extract_node_text(node: Any, source_code: bytes) -> str:
    """Extract raw UTF-8 string from a Tree-sitter AST node's byte range."""
    return source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def extract_name(node: Any, source_code: bytes) -> str:
    """Extract the identifier name of a function, method, or class node."""
    for child in node.children:
        if child.type in ("identifier", "property_identifier"):
            return extract_node_text(child, source_code)

    if node.parent and node.parent.type == "variable_declarator":
        for child in node.parent.children:
            if child.type == "identifier":
                return extract_node_text(child, source_code)

    return "anonymous"


def walk_tree_for_types(node: Any, target_types: Set[str]) -> List[Any]:
    """Recursively search AST subtree for nodes matching target node types."""
    results: List[Any] = []
    if node.type in target_types:
        results.append(node)
    for child in node.children:
        results.extend(walk_tree_for_types(child, target_types))
    return results


def get_imports_exports(root_node: Any, source_code: bytes) -> Tuple[List[str], List[str]]:
    """Extract import and export statement strings from the file root node."""
    imports: List[str] = []
    exports: List[str] = []

    import_nodes = walk_tree_for_types(root_node, {"import_statement"})
    for node in import_nodes:
        imports.append(extract_node_text(node, source_code))

    export_nodes = walk_tree_for_types(root_node, {"export_statement"})
    for node in export_nodes:
        exports.append(extract_node_text(node, source_code))

    return imports, exports


def get_calls(node: Any, source_code: bytes) -> List[str]:
    """Extract function and method invocations called from within this node."""
    call_nodes = walk_tree_for_types(node, {"call_expression"})
    calls: List[str] = []
    for c in call_nodes:
        if c.children:
            func_node = c.children[0]
            calls.append(extract_node_text(func_node, source_code))
    return calls


def normalize_code(code: str) -> str:
    """Normalize whitespace for consistent hashing across line-ending variants."""
    code = re.sub(r"\s+", " ", code)
    return code.strip()


def generate_chunk_hash(code: str) -> str:
    """Generate SHA-256 hash of normalized code content."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def generate_chunk_id(
    repo_id: str, file_path: str, symbol_name: str, symbol_type: str, chunk_hash: str
) -> str:
    """Stable chunk ID that does NOT include line numbers.

    Line numbers shift when unrelated code is inserted before a function,
    causing the VersionManager to treat an unmodified function as modified.
    Identity is strictly repo + file + symbol name + symbol type + content hash.
    """
    raw = f"{repo_id}:{file_path}:{symbol_name}:{symbol_type}:{chunk_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_javascript_file(
    file_path: str,
    source_code: bytes,
    repo_id: str = "default_repo",
    version_id: str = "v1",
) -> List[CodeChunk]:
    """Parse a JavaScript file using Tree-sitter and extract structured CodeChunks."""
    parser = get_parser()
    tree = parser.parse(source_code)
    root = tree.root_node

    file_imports, file_exports = get_imports_exports(root, source_code)

    target_types = {
        "function_declaration",
        "method_definition",
        "class_declaration",
        "arrow_function",
    }

    chunk_nodes = walk_tree_for_types(root, target_types)
    chunks: List[CodeChunk] = []

    for node in chunk_nodes:
        if node.type == "arrow_function":
            if node.parent and node.parent.type == "call_expression":
                continue

        code_text = extract_node_text(node, source_code)
        symbol_name = extract_name(node, source_code)
        symbol_type = node.type
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        calls = get_calls(node, source_code)

        node_exports: List[str] = []
        p = node.parent
        while p:
            if p.type == "export_statement":
                node_exports = [symbol_name]
                break
            p = p.parent

        norm_code = normalize_code(code_text)
        c_hash = generate_chunk_hash(norm_code)

        context_str = (
            f"File: {file_path}\n"
            f"Symbol: {symbol_name}\n"
            f"Type: {symbol_type}\n"
            f"Calls: {', '.join(calls)}\n"
            f"Code:\n{code_text}"
        )

        c_id = generate_chunk_id(repo_id, file_path, symbol_name, symbol_type, c_hash)

        chunk = CodeChunk(
            chunk_id=c_id,
            repo_id=repo_id,
            version_id=version_id,
            file_path=file_path,
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            start_line=start_line,
            end_line=end_line,
            code=code_text,
            normalized_code=norm_code,
            context=context_str,
            imports=file_imports,
            exports=node_exports,
            calls=calls,
            called_by=[],
            chunk_hash=c_hash,
        )
        chunks.append(chunk)

    return chunks
