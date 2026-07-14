"""Entry point of the project.

The extraction paths are independent, importable libraries:
  * extraction.LLM.LLM        — structured layout + LLM field extraction (Foundry)
  * extraction.ACU.Acu        — ACU client + overlap batching
  * extraction.Templates.Template — infer a template from a sample document
"""


def main():
    print("Intelli-Doc Accuracy backend. Import extraction.LLM / extraction.ACU to use.")


if __name__ == "__main__":
    main()
