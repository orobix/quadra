"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

PACKAGE_NAME = "quadra"
SKIP_FOLDERS = ["configs", "__pycache__"]


def init_file_imports(init_file: Path) -> bool:
    """Check if the __init__.py file imports anything.

    Args:
        init_file: Path to the __init__.py file

    Returns:
        True if the file imports anything, False otherwise.
    """
    with open(init_file, "r") as fd:
        return any(line.startswith("__all__") for line in fd.readlines())


def add_submodules_as_list(parent_folder: Path) -> str:
    """Add subfolders as a list to the index.md file."""
    python_files = []
    sub_folders = []
    for file in parent_folder.iterdir():
        if file.is_dir() and file.name not in SKIP_FOLDERS:
            sub_folders.append(file.name)
        elif file.suffix == ".py" and file.name != "__init__.py":
            python_files.append(file.name)
    output = []
    if len(sub_folders) > 0:
        output.append("\n## Submodules\n\n")
    for sub_folder_name in sub_folders:
        output.append(f"- [{sub_folder_name}]({sub_folder_name}/index.md)\n")
    if len(python_files) > 0:
        output.append("\n## Python Files\n\n")
    for python_file in python_files:
        python_link = python_file.replace(".py", ".md")
        output.append(f"- [{python_file}]({python_link})\n")
    return "".join(output)


nav = mkdocs_gen_files.Nav()

for path in sorted(Path(PACKAGE_NAME).rglob("*.py")):
    if any(skip_folder in path.parts for skip_folder in SKIP_FOLDERS):
        continue

    module_path = path.relative_to(".").with_suffix("")
    doc_path = path.relative_to(".").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)
    MARKDOWN_CONTENTS = ""
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
        readme_file = path.parent / "README.md"
        if readme_file.exists():
            with open(readme_file, "r") as rfd:
                MARKDOWN_CONTENTS += f"{rfd.read()}\n"
        if not init_file_imports(path):
            MARKDOWN_CONTENTS += add_submodules_as_list(path.parent)
    elif parts[-1] == "__main__":
        continue

    nav[parts] = str(doc_path)

    with mkdocs_gen_files.open(full_doc_path, "w") as mfd:
        IDENT = ".".join(parts)
        mfd.write(f"{MARKDOWN_CONTENTS}::: {IDENT}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with open(Path("README.md"), "r") as read_fd:
    readme = read_fd.read()
    readme = readme.replace("# Quadra: Deep Learning Experiment Orchestration Library", "# Home")
    readme = readme.replace("docs/", "")
    with mkdocs_gen_files.open("index.md", "w") as nav_file:  # (2)
        nav_file.write(readme)

with open("CHANGELOG.md", "r") as change_fd:
    changelog = change_fd.read()
    changelog = changelog.replace("All notable changes to this project will be documented in this file.", "")
    with mkdocs_gen_files.open("reference/CHANGELOG.md", "w") as nav_file:  # (2)
        nav_file.write(changelog)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:  # (2)
    nav_file.writelines(nav.build_literate_nav())  # (3)


if __name__ == "__main__":
    mkdocs_gen_files.__main__()
