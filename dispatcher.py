"""
Dispatcher CLI that forwards digit generation requests to backend servers
according to a mapping file.

Usage:
    python dispatcher.py [--mapping digit_mapping.json] [--output_dir generated]

Mapping file format (JSON):
    { "0": "http://host1:5000", ..., "5": "http://host2:5000", ... }

At the prompt, enter a digit (0-9) to generate an image, or 'q' to quit.
"""

import argparse
import json
import os
import requests


def load_mapping(path: str) -> dict[str, str]:
    with open(path) as f:
        mapping = json.load(f)
    missing = [str(d) for d in range(10) if str(d) not in mapping]
    if missing:
        print(f"Warning: no server mapped for digit(s): {', '.join(missing)}")
    return mapping


def request_digit(digit_str: str, mapping: dict[str, str], output_dir: str) -> None:
    if digit_str not in mapping:
        print(f"Error: digit '{digit_str}' is not in the mapping")
        return

    backend = mapping[digit_str]
    try:
        resp = requests.get(f"{backend}/generate", params={"digit": digit_str}, timeout=60)
    except requests.ConnectionError:
        print(f"Error: could not reach backend {backend}")
        return
    except requests.Timeout:
        print(f"Error: backend {backend} timed out")
        return

    if resp.status_code != 200:
        print(f"Error: backend returned {resp.status_code}: {resp.text}")
        return

    out_path = os.path.join(output_dir, f"digit_{digit_str}.png")
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"Saved -> {out_path}  (via {backend})")


def main():
    parser = argparse.ArgumentParser(description="Digit generation dispatcher")
    parser.add_argument("--mapping", type=str, default="digit_mapping.json", help="Path to digit-to-server mapping JSON")
    parser.add_argument("--output_dir", type=str, default="~/Downloads/generated", help="Directory to save generated images")
    args = parser.parse_args()

    mapping = load_mapping(args.mapping)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Digit routing:")
    for digit, server in sorted(mapping.items()):
        print(f"  {digit} -> {server}")
    print("\nEnter a digit (0-9) to generate, or 'q' to quit.")

    while True:
        try:
            user_input = input("digit> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("q", "quit", "exit"):
            break
        if not user_input:
            continue

        request_digit(user_input, mapping, args.output_dir)


if __name__ == "__main__":
    main()
