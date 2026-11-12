import argparse

from .report import run


def main():
    ap = argparse.ArgumentParser(prog="mcrisk", description="Monte Carlo portfolio risk report")
    ap.add_argument("config", nargs="?", default="portfolio.toml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--refresh", action="store_true", help="download prices again")
    args = ap.parse_args()
    run(args.config, args.out, refresh=args.refresh)


if __name__ == "__main__":
    main()
