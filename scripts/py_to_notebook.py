"""Convert scripts/tsfm_vs_classical.py (jupytext-style # %% cells) to a notebook.

Splits on `# %%` markers. A cell starting with `# %% [markdown]` becomes a markdown
cell (leading `# ` stripped); otherwise it's a code cell. Lines that are commented
shell commands (`# !uv pip ...`) are un-commented so they run as notebook magics.
"""
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "tsfm_vs_classical.py")
OUT = os.path.join(os.path.dirname(HERE), "notebooks", "tsfm_vs_classical.ipynb")


def main():
    lines = open(SRC, encoding="utf-8").read().split("\n")
    # drop the module docstring + shebang header (everything before the first # %%)
    start = next(i for i, l in enumerate(lines) if l.startswith("# %%"))
    lines = lines[start:]

    cells, buf, is_md = [], [], False

    def flush():
        if not buf:
            return
        body = "\n".join(buf).strip("\n")
        if not body.strip():
            return
        if is_md:
            txt = "\n".join(l[2:] if l.startswith("# ") else l[1:] if l == "#" else l
                            for l in body.split("\n"))
            cells.append(new_markdown_cell(txt.strip()))
        else:
            # un-comment shell magics across line continuations:
            # "# !uv pip ... \" + "#   more-pkgs" -> "!uv pip ... \" + "  more-pkgs"
            out, in_magic = [], False
            for l in body.split("\n"):
                stripped = l.lstrip()
                if stripped.startswith("# !"):
                    out.append(l.replace("# !", "!", 1))
                    in_magic = l.rstrip().endswith("\\")
                elif in_magic and stripped.startswith("#"):
                    # continuation of a shell magic — drop the leading "# "
                    out.append(l.replace("# ", "", 1) if l.startswith("# ") else l.replace("#", "", 1))
                    in_magic = l.rstrip().endswith("\\")
                else:
                    out.append(l); in_magic = False
            cells.append(new_code_cell("\n".join(out)))

    for l in lines:
        if l.startswith("# %%"):
            flush()
            buf, is_md = [], l.strip().endswith("[markdown]")
            continue
        buf.append(l)
    flush()

    nb = new_notebook(cells=cells)
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                      "accelerator": "GPU", "colab": {"provenance": []}}
    nbformat.write(nb, OUT)
    nbformat.validate(nbformat.read(OUT, as_version=4))
    print(f"wrote {OUT} · {len(cells)} cells "
          f"({sum(c.cell_type=='code' for c in cells)} code)")


if __name__ == "__main__":
    main()
