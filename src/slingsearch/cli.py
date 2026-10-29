import argparse
from .api import search


def main():
    parser = argparse.ArgumentParser(description='Retrieve question-focused evidence')
    parser.add_argument('question')
    parser.add_argument('--mode', choices=['web', 'scholar'], default='web')
    parser.add_argument('--ranker', choices=['rrf', 'jev', 'decider', 'voyage'], default='jev')
    parser.add_argument('--providers', nargs='+')
    parser.add_argument('--anchor-provider', help='Preserve this provider’s best available original result')
    parser.add_argument('--diversity', action='store_true', help='Balance shortlist coverage and diversify near relevance ties')
    parser.add_argument('--max-results', type=int, default=5)
    parser.add_argument('--token-budget', type=int, default=1500)
    parser.add_argument('--format', choices=['json', 'markdown'], default='markdown')
    args = vars(parser.parse_args())
    format = args.pop('format')
    result = search(**args)
    print(result.to_json(indent=2) if format == 'json' else result.to_markdown())
