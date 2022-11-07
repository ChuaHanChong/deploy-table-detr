"""Main executive module."""
import argparse
import requests
import pandas as pd


def main(args):
    """Run main function."""
    resp = requests.post(url=args.url, files={"file": open(args.image_path, "rb")})
    return resp.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str)
    parser.add_argument("--url", type=str, default="http://0.0.0.0:8000/predict")

    args = parser.parse_args()
    print(main(args))
