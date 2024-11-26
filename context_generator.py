import os
from pathlib import Path
import pyperclip
from rich.tree import Tree
from rich import print as rprint

def get_tree_string(directory, exclude_patterns=None):
    """Generate a string representation of the directory tree."""
    if exclude_patterns is None:
        exclude_patterns = ['.git', '__pycache__', '.pytest_cache', '.venv', 'venv']
    
    tree = Tree(f"[bold blue]{Path(directory).name}")
    
    def add_to_tree(tree, path):
        """Recursively add directory contents to tree."""
        entries = sorted(os.scandir(path), key=lambda e: (not e.is_file(), e.name.lower()))
        
        for entry in entries:
            # Skip excluded patterns
            if any(pattern in entry.path for pattern in exclude_patterns):
                continue
                
            if entry.is_file():
                tree.add(f"[green]{entry.name}")
            elif entry.is_dir():
                branch = tree.add(f"[bold yellow]{entry.name}")
                add_to_tree(branch, entry.path)
    
    add_to_tree(tree, directory)
    return tree

def read_file_contents(file_path):
    """Read and return the contents of a file with proper error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f"\n=== Contents of {file_path} ===\n" + f.read()
    except Exception as e:
        return f"\n=== Error reading {file_path}: {str(e)} ===\n"

def main():
    # Hardcode the important files to include
    important_files = [
        '.gitignore',          # Git ignore rules
        'pyproject.toml',      # Poetry configuration
        'README.md',           # Project documentation
        'app.py',             # Main Streamlit application
        'requirements.txt'     # Dependencies (if not using poetry)
    ]
    
    # Project directory (current directory by default)
    project_dir = '.'
    
    # Generate tree structure
    project_tree = get_tree_string(project_dir)
    tree_output = "\n=== Project Structure ===\n"
    
    # Capture tree output
    from io import StringIO
    tree_string = StringIO()
    rprint(project_tree, file=tree_string)
    tree_output += tree_string.getvalue()
    
    # Generate file contents for important files that exist
    files_output = ""
    for file_path in important_files:
        if os.path.exists(file_path):
            files_output += read_file_contents(file_path)
    
    # Combine outputs
    full_output = f"""Project Context Report
Generated from directory: {os.path.abspath(os.getcwd())}
{tree_output}
{files_output}
"""
    
    # Copy to clipboard
    pyperclip.copy(full_output)
    
    # Print success message and preview
    print("\nProject context has been copied to clipboard!")
    print("\nPreview of generated context:")
    print("-" * 50)
    print(full_output[:500] + "..." if len(full_output) > 500 else full_output)
    print("-" * 50)
    
    # Also print which files were actually found and included
    found_files = [f for f in important_files if os.path.exists(f)]
    print("\nFiles included in report:")
    for file in found_files:
        print(f"✓ {file}")
    
    missing_files = [f for f in important_files if not os.path.exists(f)]
    if missing_files:
        print("\nFiles not found:")
        for file in missing_files:
            print(f"✗ {file}")

if __name__ == "__main__":
    main()