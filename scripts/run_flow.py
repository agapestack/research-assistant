#!/usr/bin/env python3
"""CLI to run Prefect flows locally or trigger deployments."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_local(flow_name: str, **kwargs):
    """Run a flow locally (without Prefect server)."""
    from src.workflows import (
        collect_papers_flow,
        index_papers_flow,
        full_pipeline_flow,
    )
    from src.workflows.orchestrator import daily_update_flow, weekly_full_sync_flow

    flows = {
        "collect": collect_papers_flow,
        "index": index_papers_flow,
        "full": full_pipeline_flow,
        "daily": daily_update_flow,
        "weekly": weekly_full_sync_flow,
    }

    if flow_name not in flows:
        print(f"Unknown flow: {flow_name}")
        print(f"Available: {', '.join(flows.keys())}")
        sys.exit(1)

    flow_fn = flows[flow_name]
    print(f"Running {flow_name} flow locally...")
    result = flow_fn(**kwargs)
    print(f"\nResult: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run Prefect flows")
    parser.add_argument(
        "flow",
        choices=["collect", "index", "full", "daily", "weekly"],
        help="Flow to run",
    )
    parser.add_argument("--days", type=int, default=30, help="Days back to search")
    parser.add_argument("--max-per-query", type=int, default=100, help="Max results per query")
    parser.add_argument("--limit", type=int, default=50, help="Papers to index")

    args = parser.parse_args()

    kwargs = {}
    if args.flow == "collect":
        kwargs = {
            "max_per_query": args.max_per_query,
            "days_back": args.days,
        }
    elif args.flow == "index":
        kwargs = {"limit": args.limit}
    elif args.flow == "full":
        kwargs = {
            "days_back": args.days,
            "max_per_query": args.max_per_query,
            "index_limit": args.limit,
        }

    run_local(args.flow, **kwargs)


if __name__ == "__main__":
    main()
