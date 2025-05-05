import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=str)
    parser.add_argument("num_trials", type=int)
    parser.add_argument("filename", type=str)
    args = parser.parse_args()

    for i in range(args.num_trials):
        print(f"=== trial {i} ===")
        try:
            os.system(f"python3 -m scripts.{args.script} >> {args.filename}")
        except KeyboardInterrupt:
            sys.exit(1)
    print("Done!")

if __name__ == "__main__":
    main()