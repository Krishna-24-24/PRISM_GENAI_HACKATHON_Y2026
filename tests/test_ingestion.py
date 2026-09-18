import os
import tempfile
from src.ingestion.scanner import discover_js_files

def test_discover_js_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "node_modules", "package"))
        os.makedirs(os.path.join(tmpdir, "src"))
        
        with open(os.path.join(tmpdir, "src", "index.js"), "w") as f:
            f.write("console.log('test');")
            
        with open(os.path.join(tmpdir, "src", "data.json"), "w") as f:
            f.write("{}")
            
        with open(os.path.join(tmpdir, "node_modules", "package", "index.js"), "w") as f:
            f.write("module.exports = {};")
            
        with open(os.path.join(tmpdir, "src", "bundle.min.js"), "w") as f:
            f.write("minified")
            
        files = discover_js_files(tmpdir)
        
        assert len(files) == 1
        assert files[0].endswith("index.js")
