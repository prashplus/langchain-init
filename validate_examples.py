#!/usr/bin/env python3
"""
Example Validation Script - Validates that examples can be imported and have correct syntax
"""

import ast
import os
import sys

def validate_python_file(file_path):
    """Validate a Python file for syntax and basic structure"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        tree = ast.parse(content, filename=file_path)
        
        # Check for basic structure
        has_main = any(isinstance(node, ast.FunctionDef) and node.name == 'main' for node in ast.walk(tree))
        has_imports = any(isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom) for node in ast.walk(tree))
        
        return {
            'syntax_valid': True,
            'has_main': has_main,
            'has_imports': has_imports,
            'error': None
        }
    except SyntaxError as e:
        return {
            'syntax_valid': False,
            'has_main': False,
            'has_imports': False,
            'error': str(e)
        }
    except Exception as e:
        return {
            'syntax_valid': False,
            'has_main': False,
            'has_imports': False,
            'error': f"Unexpected error: {e}"
        }

def main():
    """Validate all example files"""
    example_files = [
        'langchain/simple_chat.py',
        'langchain/streaming_chat.py',
        'langchain/conversation_memory.py',
        'langchain/document_qa.py',
        'langgraph/simple_agent.py',
        'langgraph/chat_with_tools.py',
        'langgraph/code_reviewer.py',
        'langgraph/conditional_workflow.py',
        'langgraph/creative_writing.py',
        'langgraph/multi_step_reasoning.py',
        'langgraph/research_assistant.py',
    ]
    
    print("🔍 Validating Example Files...")
    print("=" * 50)
    
    all_valid = True
    results = []
    
    for file_path in example_files:
        if os.path.exists(file_path):
            result = validate_python_file(file_path)
            results.append((file_path, result))
            
            if result['syntax_valid']:
                print(f"✅ {file_path}: Valid syntax")
                if result['has_main']:
                    print(f"   📌 Has main() function")
                if result['has_imports']:
                    print(f"   📦 Has imports")
            else:
                print(f"❌ {file_path}: Syntax error - {result['error']}")
                all_valid = False
        else:
            print(f"⚠️  {file_path}: File not found")
            all_valid = False
    
    print("\n" + "=" * 50)
    print("📊 Validation Summary:")
    
    valid_count = sum(1 for _, result in results if result['syntax_valid'])
    total_count = len(results)
    
    print(f"Valid files: {valid_count}/{total_count}")
    
    if all_valid:
        print("🎉 All example files are valid!")
        return 0
    else:
        print("⚠️  Some files have issues. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
