import os
from typing import List

IGNORED_DIRS = {
    "node_modules", ".git", "dist", "build", "coverage", ".next", ".nuxt", "out"
}

IGNORED_EXTENSIONS = {
    ".min.js", ".bundle.js", ".map"
}

ALLOWED_EXTENSIONS = {
    ".js", ".jsx", ".mjs", ".cjs"
}

def discover_js_files(root_path: str) -> List[str]:
    """Recursively discover JavaScript source files while ignoring irrelevant directories and files."""
    found_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Mutate dirnames in-place to avoid walking into ignored directories
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in ALLOWED_EXTENSIONS and not any(filename.endswith(ign_ext) for ign_ext in IGNORED_EXTENSIONS):
                filepath = os.path.join(dirpath, filename)
                # Security: Skip files larger than 1MB to prevent DoS via memory exhaustion during AST parsing
                try:
                    if os.path.getsize(filepath) <= 1024 * 1024:
                        found_files.append(filepath)
                except OSError:
                    pass
                
    return sorted(found_files)
