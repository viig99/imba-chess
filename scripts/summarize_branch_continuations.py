"""Postprocess frozen branching games; no model execution or training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import chess
import chess.pgn

from imba_chess.data.self_play_store import atomic_json
from imba_chess.self_play.collector import terminal_outcome
from imba_chess.self_play.seeds import file_hash
from scripts.audit_branch_continuations import ARMS, board_for, root_score, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory
    manifest = json.loads((directory / 'manifest.json').read_text())
    ident = manifest['identity']
    positions = ident['positions']
    game_paths = sorted((directory / 'games').glob('*.json'))
    games = [json.loads(p.read_text()) for p in game_paths]
    values = {r['position_id']: r for r in json.loads((directory / 'cached_actor300_values.json').read_text())}
    vocab = json.loads(Path('artifacts/move_vocab_static_uci.json').read_text())['token_to_id']
    id_to_uci = {i: u for u, i in vocab.items()}
    pgns = []
    for g in games:
        board = board_for(g['prefix_moves'] + g['moves'])
        if g['status'] == 'completed':
            terminal = terminal_outcome(board)
            assert terminal is not None and terminal[0] == g['outcome_white']
        else:
            assert g['outcome_white'] is None
        game = chess.pgn.Game.from_board(board)
        game.headers.update(Event='Frozen actor300 branching diagnostic', White='actor300', Black='actor300',
                            Site=g['game_id'], Result=board.result(claim_draw=True) if g['status'] == 'completed' else '*')
        pgns.append(str(game))
    (directory / 'games.pgn').write_text('\n\n'.join(pgns) + '\n')
    result = summarize(positions, games, ident['repeats'], ident['seed'])
    result['postprocess_source_sha256'] = file_hash(__file__)
    result['game_file_sha256'] = {p.name: file_hash(p) for p in game_paths}
    analysis = []
    for p in positions:
        pid = p['position_id']
        labels = manifest['reply_labels'][pid]
        gs = [g for g in games if g['position_id'] == pid]
        row = dict(position_id=pid, source=p['source'], fen=p['fen'], bad=p['bad'], good=p['good'],
                   engine_reply=labels['best_reply'], good_replies=labels['good_replies'],
                   cached_bad=p['stockfish_bad']['expectation'], cached_good=p['stockfish_good']['expectation'],
                   rechecked_bad=1-labels['scores'][labels['best_reply']]['expectation'], **{k:v for k,v in values[pid].items() if k != "position_id"})
        row['label_shift'] = row['rechecked_bad'] - row['cached_bad']
        for arm in ARMS:
            subset = [g for g in gs if g['arm'] == arm]
            scores = [root_score(g, p['root_white']) for g in subset if g['status'] == 'completed']
            row[arm] = dict(scores=scores, mean=statistics.mean(scores) if scores else None,
                            unique_continuations=len({tuple(g['moves']) for g in subset}),
                            first_moves=dict(Counter(g['moves'][0] for g in subset if g['moves'])))
        natural = [g for g in gs if g['arm'] == 'bad']
        row['reply_traces'] = []
        for g in natural:
            if not g['targets']:
                continue
            t = g['targets'][0]
            moves = [id_to_uci[i] for i in t['legal_ids']]
            ranks = {moves[i]: rank+1 for rank, i in enumerate(sorted(range(len(moves)), key=lambda i: -t['root_log_priors'][i]))}
            row['reply_traces'].append(dict(game_id=g['game_id'], chosen=t['move_uci'],
                chosen_good=g['first_reply_good'], root_player_score=root_score(g, p['root_white']),
                good_reply_best_prior_rank=min(ranks[m] for m in labels['good_replies']),
                good_reply_visits=sum(t['visits'][moves.index(m)] for m in labels['good_replies']),
                chosen_q=t['qvalues'][moves.index(t['move_uci'])],
                good_reply_q={m:t['qvalues'][moves.index(m)] for m in labels['good_replies'] if t['visits'][moves.index(m)] > 0}))
        analysis.append(row)
    result['details'] = analysis
    traces = [t for r in analysis for t in r['reply_traces']]
    missed = [t for t in traces if not t['chosen_good']]
    result['reply_search_diagnostics'] = dict(
        traced=len(traces), missed=len(missed),
        good_reply_prior_rank_median=statistics.median(t['good_reply_best_prior_rank'] for t in traces) if traces else None,
        missed_despite_visiting_good_reply=sum(t['good_reply_visits'] > 0 for t in missed),
        missed_without_visiting_good_reply=sum(t['good_reply_visits'] == 0 for t in missed),
        missed_with_good_reply_higher_q=sum(t['good_reply_visits'] > 0 and max(t['good_reply_q'].values()) > t['chosen_q'] for t in missed))
    # The original engine labels shifted materially for some positions: sensitivity analysis.
    stable_ids = {r['position_id'] for r in analysis if abs(r['label_shift']) <= .1}
    wrong_ids = {r['position_id'] for r in analysis if r['main_prefers_bad']}
    for key, ids in [('stable_engine_labels', stable_ids), ('actor300_still_misorders', wrong_ids)]:
        result[key] = summarize([p for p in positions if p['position_id'] in ids],
                               [g for g in games if g['position_id'] in ids], ident['repeats'], ident['seed'])
    result['reply_conditional'] = {}
    for good in (True, False):
        subset = [g for g in games if g['arm'] == 'bad' and g['first_reply_good'] is good]
        scores = [root_score(g, g['root_white']) for g in subset if g['status'] == 'completed']
        result['reply_conditional'][str(good)] = dict(games=len(scores), root_player_score=statistics.mean(scores) if scores else None,
            wins=scores.count(1), draws=scores.count(.5), losses=scores.count(0))
    lines = ["# Actor300 branching diagnostic", "",
        f"Completed {len(games) - result['incomplete']}/{len(games)} saved games; planned {len(positions)*ident['repeats']*len(ARMS)}.",
        "Outcomes are from the original parent player's perspective. No training was performed.", "",
        "| Arm | W | D | L | Score |", "|---|---:|---:|---:|---:|"]
    for arm, r in result['arms'].items():
        score = f"{100*r['score']:.1f}%" if r['score'] is not None else "pending"
        lines.append(f"| {arm} | {r['wins']} | {r['draws']} | {r['losses']} | {score} |")
    lines += ["", "Paired contrasts require all four completed outcomes in both arms at a position."]
    for name, r in result['contrasts'].items():
        lo, hi = r['source_game_bootstrap_95ci']
        lines.append(f"- {name}: {100*r['mean']:+.1f} points; source-game bootstrap 95% interval [{100*lo:+.1f}, {100*hi:+.1f}]; {r['positions']} positions.")
    lines += ["", f"Natural bad-arm first replies within 2 points of Stockfish's best: {result['first_reply_good']}/{result['first_reply_observed']}.", "",
        "| Position | Main bad/good | Rollout bad/good/forced | Good replies found |", "|---|---|---|---|"]
    for r in analysis:
        means = [f"{r[a]['mean']:.3f}" if r[a]['mean'] is not None else "pending" for a in ARMS]
        found = sum(t['chosen_good'] for t in r['reply_traces'])
        lines.append(f"| {r['position_id']} | {r['main_bad']:.3f}/{r['main_good']:.3f} | {' / '.join(means)} | {found}/{len(r['reply_traces'])} |")
    lines += ["", "Interpretation limits:",
        "- These are selected historical baseline blunders, not an independent validation set; actor300 still misorders eight of twelve before this experiment.",
        "- Four continuations per arm estimate this frozen noisy playing procedure, not perfect-play value.",
        "- The forced-reply arm is an intervention. First-reply quality does not establish execution of a whole tactical refutation.",
        "- Engine labels are finite-budget proxies; analysis.json includes sensitivity to labels shifting by over 0.10 when rescored after the move.",
        "- Outcomes can fail to separate moves because either player makes later mistakes. No training benefit has been tested.",
        "", "See PROTOCOL.md, analysis.json, games.pgn and individual game JSON files for exact histories, labels and search traces."]
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    atomic_json(directory / 'analysis.json', result)
    print(json.dumps({k:v for k,v in result.items() if k in ['arms','contrasts','reply_conditional','first_reply_good','first_reply_observed','incomplete']}, indent=2))


if __name__ == '__main__':
    main()
