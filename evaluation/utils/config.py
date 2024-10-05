from pathlib import Path


def get_project_root():
    current_dir = Path(__file__).resolve().parent  # For scripts
    while not (current_dir / "README.md").exists():  # or check for another root file
        if current_dir.parent == current_dir:  # reached the root of the file system
            raise Exception("Project root not found.")
        current_dir = current_dir.parent
    return str(current_dir)


def get_project_root_nb():
    current_dir = Path.cwd()  # For notebooks
    while not (current_dir / "README.md").exists():  # or check for another root file
        if current_dir.parent == current_dir:  # reached the root of the file system
            raise Exception("Project root not found.")
        current_dir = current_dir.parent
    return str(current_dir)
