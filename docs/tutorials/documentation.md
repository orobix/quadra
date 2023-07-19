# Build documentation

## Creating the Website with updated content.


there are multiple ways to create the website.

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build -d <path>` - Build the documentation site to output path.
* `mkdocs -h` - Print help message and exit.

!!! warning
    The website generator creates the pages for each code reference (python file) automatically.
    The user does not need to create the pages manually. However, other pages such as tutorials or explanations should
    be created manually.

## Contributing

If you want to create a new page, follow the given steps:

1. Under the docs folder create or use related folder for each markdown file.
2. Open `mkdocs.yml` and add the new page under the `nav` tag with a relative path to `docs` folder.

```yaml
nav:
  - index.md
  # defer to gen-files + literate-nav
  - Code Reference: reference/
  - Tutorials:
      - Getting Started: tutorials/install.md
  # You can add new pages here
  # ...
```

## Developing Landing Page


Landing page extends the `home.html` from `mkdocs-material` theme. It uses `tailwindcss` for styling. The developer has to run the following command to update the `tailwindcss` styles.

```bash
npx tailwindcss -i ./docs/template/landing.css -o ./docs/css/landing.css --watch
```

## Features

Most of the available features are defined [here](https://squidfunk.github.io/mkdocs-material/reference/).
