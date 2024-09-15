from app import main
import argparse

parser = argparse.ArgumentParser(description="Run a psychology chatbot model.")
parser.add_argument(
    "-prompt",
    type=str,
    help="The input prompt to send to the model for processing."
)

args = parser.parse_args()
result = main(args.prompt)
print(result)