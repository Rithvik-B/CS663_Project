"""
Helper script to set up data directories with links to parent repo data.
"""

import os
import sys
from pathlib import Path


def setup_data_links():
    """Create symbolic links or copies to parent data directories."""
    project_root = Path(__file__).parent
    parent_data = project_root.parent / "data"
    
    content_source = parent_data / "content-images"
    style_source = parent_data / "style-images"
    
    content_dest = project_root / "data" / "content_examples"
    style_dest = project_root / "data" / "style_examples"
    
    # Create destination directories
    content_dest.mkdir(parents=True, exist_ok=True)
    style_dest.mkdir(parents=True, exist_ok=True)
    
    # Try to create symbolic links (works on Unix/Mac)
    # On Windows, will need to use junctions or copies
    if sys.platform == "win32":
        print("Windows detected. Creating junctions or copies...")
        # Use junctions on Windows
        try:
            import subprocess
            if content_source.exists():
                subprocess.run(
                    ["mklink", "/J", str(content_dest), str(content_source)],
                    shell=True, check=False
                )
            if style_source.exists():
                subprocess.run(
                    ["mklink", "/J", str(style_dest), str(style_source)],
                    shell=True, check=False
                )
            print("Junctions created (or already exist)")
        except:
            print("Could not create junctions. Please manually copy or link:")
            print(f"  {content_source} -> {content_dest}")
            print(f"  {style_source} -> {style_dest}")
    else:
        # Unix/Mac: use symbolic links
        if content_source.exists() and not (content_dest / ".linked").exists():
            if content_dest.exists():
                # Remove if it's not a link
                if not content_dest.is_symlink():
                    print(f"Removing existing {content_dest}")
                    import shutil
                    shutil.rmtree(content_dest)
            os.symlink(content_source, content_dest)
            print(f"Created link: {content_dest} -> {content_source}")
        
        if style_source.exists() and not (style_dest / ".linked").exists():
            if style_dest.exists():
                if not style_dest.is_symlink():
                    print(f"Removing existing {style_dest}")
                    import shutil
                    shutil.rmtree(style_dest)
            os.symlink(style_source, style_dest)
            print(f"Created link: {style_dest} -> {style_source}")
    
    print("\nData setup complete!")
    print(f"Content images: {content_dest}")
    print(f"Style images: {style_dest}")


if __name__ == "__main__":
    setup_data_links()

