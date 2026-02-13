#!/usr/bin/env python3
"""
Verification script for Lab-4 implementation
Checks that all required files exist and have proper structure
"""

import os
import json
import sys

def check_file_exists(filepath, file_type="file"):
    """Check if file exists and print status"""
    exists = os.path.exists(filepath)
    symbol = "✓" if exists else "✗"
    print(f"{symbol} {file_type}: {filepath}")
    return exists

def verify_lab4():
    """Verify Lab-4 implementation"""
    print("="*60)
    print("Lab-4 Implementation Verification")
    print("="*60)
    
    all_good = True
    
    # Check directory structure
    print("\n1. Directory Structure:")
    all_good &= check_file_exists("Lab-4", "Directory")
    
    # Check required files
    print("\n2. Required Files:")
    required_files = [
        "Lab-4/Lab-4.ipynb",
        "Lab-4/README.md",
        "Lab-4/IMPLEMENTATION_SUMMARY.md",
        "Lab-4/QUICK_START.md",
        "requirements.txt",
        "README.md"
    ]
    
    for filepath in required_files:
        all_good &= check_file_exists(filepath)
    
    # Check notebook structure
    print("\n3. Notebook Structure:")
    try:
        with open("Lab-4/Lab-4.ipynb", 'r') as f:
            nb = json.load(f)
        
        print(f"✓ Notebook JSON valid")
        print(f"  - Total cells: {len(nb['cells'])}")
        print(f"  - Code cells: {sum(1 for c in nb['cells'] if c['cell_type'] == 'code')}")
        print(f"  - Markdown cells: {sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')}")
        
        # Check for all 7 problem statements
        content = json.dumps(nb)
        print("\n4. Problem Statements:")
        for i in range(1, 8):
            found = f"Problem Statement {i}" in content
            symbol = "✓" if found else "✗"
            print(f"{symbol} Problem Statement {i}")
            all_good &= found
            
    except Exception as e:
        print(f"✗ Error reading notebook: {e}")
        all_good = False
    
    # Check README updates
    print("\n5. Documentation:")
    try:
        with open("README.md", 'r') as f:
            readme = f.read()
        
        has_lab4 = "Lab-4" in readme or "Lab 4" in readme
        symbol = "✓" if has_lab4 else "✗"
        print(f"{symbol} Main README includes Lab-4")
        all_good &= has_lab4
        
    except Exception as e:
        print(f"✗ Error reading README: {e}")
        all_good = False
    
    # Check requirements
    print("\n6. Dependencies:")
    try:
        with open("requirements.txt", 'r') as f:
            reqs = f.read()
        
        required_deps = [
            "torch>=2.6.0",
            "imbalanced-learn>=0.8.0",
            "umap-learn>=0.5.0",
            "Pillow>=10.2.0"
        ]
        
        for dep in required_deps:
            base_dep = dep.split(">=")[0]
            found = base_dep in reqs
            symbol = "✓" if found else "✗"
            print(f"{symbol} {dep}")
            all_good &= found
            
    except Exception as e:
        print(f"✗ Error reading requirements: {e}")
        all_good = False
    
    # Final summary
    print("\n" + "="*60)
    if all_good:
        print("✅ ALL CHECKS PASSED - Lab-4 implementation complete!")
    else:
        print("❌ SOME CHECKS FAILED - please review issues above")
    print("="*60)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(verify_lab4())
