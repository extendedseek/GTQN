from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from gtqn.envs.demand_generator import FlowSpec, write_routes_xml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--flows_yaml", type=str, required=True)
    args = ap.parse_args()

    data = yaml.safe_load(Path(args.flows_yaml).read_text(encoding="utf-8"))
    flows = [FlowSpec(**f) for f in data["flows"]]
    write_routes_xml(args.out, flows)

if __name__ == "__main__":
    main()
