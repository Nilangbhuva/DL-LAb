#!/usr/bin/env python3
"""
Verification script for Lab-3 notebook
This script checks that the notebook is properly structured without running the code.
"""

import json
import sys

def verify_notebook(notebook_path):
    """Verify the Lab-3 notebook structure"""
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        print("="*70)
        print("Lab-3 Notebook Verification Report")
        print("="*70)
        
        # Check basic structure
        assert 'cells' in notebook, "Notebook missing 'cells' key"
        assert 'metadata' in notebook, "Notebook missing 'metadata' key"
        assert notebook['nbformat'] == 4, f"Expected nbformat 4, got {notebook['nbformat']}"
        
        cells = notebook['cells']
        print(f"\n✅ Valid Jupyter Notebook (format {notebook['nbformat']}.{notebook['nbformat_minor']})")
        print(f"✅ Total cells: {len(cells)}")
        
        # Count cell types
        markdown_cells = sum(1 for c in cells if c['cell_type'] == 'markdown')
        code_cells = sum(1 for c in cells if c['cell_type'] == 'code')
        print(f"   - Markdown cells: {markdown_cells}")
        print(f"   - Code cells: {code_cells}")
        
        # Check for required sections
        required_sections = [
            'Dataset Preparation',
            'CNN Architecture',
            'LeNet',
            'AlexNet',
            'VGGNet',
            'ResNet',
            'Loss Functions',
            'Focal Loss',
            'ArcFace',
            'Training',
            't-SNE'
        ]
        
        notebook_text = ' '.join(
            ' '.join(cell['source']) 
            for cell in cells 
            if cell['source']
        )
        
        print(f"\n{'Section':<30} {'Found':<10}")
        print("-" * 40)
        missing_sections = []
        for section in required_sections:
            found = section in notebook_text
            status = "✅" if found else "❌"
            print(f"{section:<30} {status:<10}")
            if not found:
                missing_sections.append(section)
        
        # Check for architecture implementations
        architectures = [
            'LeNet5',
            'AlexNet',
            'VGGNet',
            'ResNet50',
            'ResNet100',
            'EfficientNet',
            'InceptionV3',
            'MobileNet'
        ]
        
        print(f"\n{'Architecture':<30} {'Implemented':<10}")
        print("-" * 40)
        missing_architectures = []
        for arch in architectures:
            found = f'class {arch}' in notebook_text or f'{arch}(' in notebook_text
            status = "✅" if found else "❌"
            print(f"{arch:<30} {status:<10}")
            if not found:
                missing_architectures.append(arch)
        
        # Check for loss functions
        loss_functions = [
            'BCELossMultiClass',
            'FocalLoss',
            'ArcFaceLoss'
        ]
        
        print(f"\n{'Loss Function':<30} {'Implemented':<10}")
        print("-" * 40)
        missing_loss = []
        for loss in loss_functions:
            found = f'class {loss}' in notebook_text
            status = "✅" if found else "❌"
            print(f"{loss:<30} {status:<10}")
            if not found:
                missing_loss.append(loss)
        
        # Summary
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        
        if missing_sections:
            print(f"⚠️  Missing sections: {', '.join(missing_sections)}")
        if missing_architectures:
            print(f"⚠️  Missing architectures: {', '.join(missing_architectures)}")
        if missing_loss:
            print(f"⚠️  Missing loss functions: {', '.join(missing_loss)}")
        
        if not (missing_sections or missing_architectures or missing_loss):
            print("✅ All required components are present!")
            print("✅ Notebook structure verification PASSED")
            return True
        else:
            print("⚠️  Some components are missing")
            print("⚠️  Notebook structure verification FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Error reading notebook: {e}")
        return False

if __name__ == "__main__":
    notebook_path = "Lab-3/Lab-3.ipynb"
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
    
    success = verify_notebook(notebook_path)
    sys.exit(0 if success else 1)
