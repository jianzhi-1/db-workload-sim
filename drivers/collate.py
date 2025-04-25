import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=str)
    parser.add_argument("num_trials", type=int)
    parser.add_argument("--output", default="output.txt")
    parser.add_argument("--workload", choices=["smallbank", "tpcc"], default="smallbank", help="Workload type to use")
    args = parser.parse_args()

    res = []

    for i in range(args.num_trials):
        print(f"=== trial {i} ===")
        #stream = os.popen(f"python -m ./scripts/script_{args.script}.py")
        #stream = os.popen(f"python -m {args.script}")
        stream = os.popen(f"python -m scripts.{args.script} --workload {args.workload}")
        output = stream.read()
        res.append(eval(output))
    
    with open(args.output, "w") as f:
        json.dump(res, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    main()