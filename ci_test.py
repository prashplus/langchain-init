#!/usr/bin/env python3
"""
CI Test Script - Tests that can run in CI without external dependencies like Ollama
"""

import sys
import os
import importlib.util

def test_file_structure():
    """Test that all expected files exist"""
    expected_files = [
        'README.md',
        'test_dependencies.py',
        'langchain/README.md',
        'langchain/requirements.txt',
        'langchain/simple_chat.py',
        'langchain/streaming_chat.py',
        'langchain/conversation_memory.py',
        'langchain/document_qa.py',
        'langchain/test_setup.py',
        'langgraph/README.md',
        'langgraph/requirements.txt',
        'langgraph/simple_agent.py',
        'langgraph/chat_with_tools.py',
        'langgraph/test_setup.py',
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All expected files exist")
    return True

def test_python_syntax():
    """Test that all Python files have valid syntax"""
    python_files = []
    
    # Find all Python files
    for root, dirs, files in os.walk('.'):
        # Skip .git and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            compile(content, file_path, 'exec')
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
    
    if syntax_errors:
        print(f"❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"   {error}")
        return False
    
    print(f"✅ All {len(python_files)} Python files have valid syntax")
    return True

def test_imports_without_execution():
    """Test that imports work without executing code that requires external services"""
    test_cases = [
        {
            'name': 'LangChain Core',
            'imports': [
                'from langchain_core.prompts import ChatPromptTemplate',
                'from langchain_core.output_parsers import StrOutputParser',
            ]
        },
        {
            'name': 'LangChain Community (basic)',
            'imports': [
                'from langchain_community.llms import Ollama',
            ]
        }
    ]
    
    failed_tests = []
    for test_case in test_cases:
        try:
            for import_statement in test_case['imports']:
                exec(import_statement)
            print(f"✅ {test_case['name']} imports successful")
        except ImportError as e:
            failed_tests.append(f"{test_case['name']}: {e}")
    
    if failed_tests:
        print(f"❌ Import errors:")
        for error in failed_tests:
            print(f"   {error}")
        return False
    
    return True

def test_module_loading():
    """Test that our modules can be loaded without execution"""
    modules_to_test = [
        'langchain.simple_chat',
        'langchain.test_setup',
        'langgraph.test_setup',
        'test_dependencies',
    ]
    
    failed_modules = []
    for module_name in modules_to_test:
        try:
            # Convert module path to file path
            file_path = module_name.replace('.', os.sep) + '.py'
            if os.path.exists(file_path):
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is not None:
                    module = importlib.util.module_from_spec(spec)
                    # We won't execute the module, just load it
                    print(f"✅ Module {module_name} can be loaded")
                else:
                    failed_modules.append(f"{module_name}: Could not create module spec")
            else:
                failed_modules.append(f"{module_name}: File not found")
        except Exception as e:
            failed_modules.append(f"{module_name}: {e}")
    
    if failed_modules:
        print(f"❌ Module loading errors:")
        for error in failed_modules:
            print(f"   {error}")
        return False
    
    return True

def main():
    """Run all CI tests"""
    print("🧪 Running CI Tests...")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("Import Tests", test_imports_without_execution),
        ("Module Loading", test_module_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 CI Test Results:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All CI tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
