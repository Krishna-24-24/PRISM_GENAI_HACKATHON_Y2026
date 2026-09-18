import pytest
from src.parsing.parser import parse_javascript_file
from src.chunking.models import CodeChunk

JS_CODE = b'''
import { something } from 'module';

export class Calculator {
    constructor() {
        this.value = 0;
    }
    
    add(x) {
        this.value += x;
        return this.value;
    }
}

function process(data) {
    console.log(data);
    const cb = () => { return data.map(x => x * 2); };
    return cb();
}

export const multiply = (a, b) => a * b;
'''

def test_parse_javascript_file():
    chunks = parse_javascript_file("test.js", JS_CODE)
    
    assert len(chunks) >= 4
    
    names = [c.symbol_name for c in chunks]
    assert "Calculator" in names
    assert "add" in names
    assert "process" in names
    assert "multiply" in names
    
    calc_chunk = next(c for c in chunks if c.symbol_name == "Calculator")
    assert calc_chunk.symbol_type == "class_declaration"
    assert "import { something } from 'module';" in calc_chunk.imports
    
    process_chunk = next(c for c in chunks if c.symbol_name == "process")
    assert "console.log" in process_chunk.calls
    
    multiply_chunk = next(c for c in chunks if c.symbol_name == "multiply")
    assert multiply_chunk.exports == ["multiply"]
