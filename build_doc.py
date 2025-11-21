import pdoc
import pathlib


def main():
    pdoc.render.configure(math=True, docformat="numpy")

    pdoc.pdoc(r"./multipar_bayes", output_directory=pathlib.Path("./doc"))


if __name__ == "__main__":
    main()
