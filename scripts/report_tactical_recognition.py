"""Render the frozen tactical-recognition output as tables, CSV and annotated PGN."""
import argparse
from collections import Counter
import csv
import html
import json
from pathlib import Path

import chess
import chess.pgn

from imba_chess.data.self_play_store import atomic_json
from imba_chess.self_play.seeds import file_hash


def percent(value):
    return '—' if value is None else f'{100 * value:.1f}%'


def estimate(metrics, key):
    item = metrics.get(key, {})
    if item.get('mean') is None:
        return '—'
    lo, hi = item['ci95']
    return f'{100 * item["mean"]:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]'


def render(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows, metrics = summary['positions_detail'], summary['metrics']
    lines, predictions = {}, {}
    flat = []
    for pos in manifest['positions']:
        for arm in ('bad', 'good'):
            stem = f'{pos["position_id"]}-{arm}'
            line = json.loads((directory / 'lines' / f'{stem}.json').read_text())
            lines[pos['position_id'], arm] = line
            pred = {name: json.loads((directory / name / f'{stem}.json').read_text())['states']
                    for name in ('actor300', 'start')}
            predictions[pos['position_id'], arm] = pred
            for s in line['states']:
                i = s['ply']
                flat.append(dict(position_id=pos['position_id'], source=pos['source'], arm=arm,
                    ply=i, move=s['move'], fen=s['fen'], root_white=pos['root_white'],
                    turn_white=s['turn_white'], material=s['material'], in_check=s['in_check'],
                    terminal=s['terminal']['termination'] if s['terminal'] else '',
                    resolution=';'.join(s.get('resolution', [])),
                    engine_100k=s.get('engine', {}).get('expectation'),
                    engine_1m=s.get('verified', {}).get('expectation'),
                    cp=s.get('verified', {}).get('cp'), mate=s.get('verified', {}).get('mate'),
                    actor_main=pred['actor300'][i]['main'], actor_aux=pred['actor300'][i]['aux'],
                    start_main=pred['start'][i]['main'], start_aux=pred['start'][i]['aux']))
    with (directory / 'positions.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    with (directory / 'continuations.pgn').open('w') as f:
        for (pid, arm), line in lines.items():
            game = chess.pgn.Game()
            game.headers.update(Event='Frozen tactical recognition', Site='Local diagnostic',
                                Date='2026.09.24', White='Forced branch / Stockfish 18',
                                Black='Forced branch / Stockfish 18', Round=f'{pid}-{arm}')
            game.headers['SourceGame'] = line['source']
            game.headers['PositionId'] = pid
            game.headers['Branch'] = arm
            node = game
            for uci in line['prefix']:
                node = node.add_variation(chess.Move.from_uci(uci))
            pred = predictions[pid, arm]
            for s in line['states']:
                i = s['ply']
                if s['move']:
                    node = node.add_variation(chess.Move.from_uci(s['move']))
                node.comment = (f'Original-player perspective. Ply {i}. '
                    f'SF {percent(s.get("verified", {}).get("expectation"))}; '
                    f'actor main {percent(pred["actor300"][i]["main"])}; '
                    f'actor aux {percent(pred["actor300"][i]["aux"])}; '
                    f'start main {percent(pred["start"][i]["main"])}. '
                    f'{", ".join(s.get("resolution", []))}')
            if line['states'][-1]['terminal']:
                terminal = line['states'][-1]['terminal']
                score = terminal['expectation']
                white_score = score if line['root_white'] else 1 - score
                game.headers['Result'] = '1/2-1/2' if score == .5 else ('1-0' if white_score == 1 else '0-1')
            print(game, file=f, end='\n\n')

    report = ['# Frozen tactical recognition: actor300 versus starting flattened checkpoint', '',
        f'Completed {summary["positions"]}/{summary["expected_positions"]} audited positions '
        f'from {summary["source_games"]} source games, with both bad and good branches.', '',
        f'Physical endpoints: {summary["endpoints"]["physical"]}; '
        f'settled material loss: {summary["endpoints"]["settled_material_loss"]}; '
        f'opponent promotion: {summary["endpoints"]["opponent_promotion"]}; '
        f'confirmed forced-mate endpoints: {summary["endpoints"]["forced_mate"]}; '
        f'no qualifying endpoint: {summary["unresolved"]}. Categories can overlap.', '',
        'This is a selected historical baseline-blunder set, not a representative position sample. '
        'Stockfish WDL is a reference for stronger continuation play, not the empirical win rate of our actor.', '',
        '## Error at independently identified physical consequences', '',
        'All entries are percentage points with source-game bootstrap 95% intervals. '
        'Positive actor-minus-start error means actor300 is worse.', '',
        '| Metric | Starting main | Actor300 main | Actor300 auxiliary |',
        '|---|---:|---:|---:|']
    for title, suffix in [('Mean absolute error', 'error'), ('Mean signed overestimate', 'bias'),
                           ('Still predicts advantage: percent of cases', 'wrong_advantage')]:
        report.append('| ' + title + ' | ' + ' | '.join(estimate(metrics, f'{name}_{head}_physical_{suffix}')
                      for name, head in [('start', 'main'), ('actor300', 'main'), ('actor300', 'aux')]) + ' |')
    report += ['', 'Paired actor-minus-start main error at physical endpoints: '
               + estimate(metrics, 'actor_minus_start_main_physical_error') + ' points.',
               'Actor main error change from immediately after the blunder to the physical endpoint: '
               + estimate(metrics, 'actor300_main_physical_error_change') + ' points.',
               'Good-alternative control error at the same ply (where still nonterminal): '
               + estimate(metrics, 'actor300_main_good_control_error') + ' actor300; '
               + estimate(metrics, 'start_main_good_control_error') + ' starting checkpoint.',
               'Actor300 main remains unrecognized at the end of observation: '
               + estimate(metrics, 'actor300_main_unrecognized_at_end') + ' percent of physical-endpoint cases.', '',
               '## Confirmed forced-mate positions (not physical-loss endpoints)', '',
               '| Readout | Absolute error | Still predicts advantage: percent of cases |',
               '|---|---:|---:|']
    for name, head in [('start', 'main'), ('actor300', 'main'), ('actor300', 'aux')]:
        report.append(f'| {name} {head} | {estimate(metrics, f"{name}_{head}_forced_mate_error")} | '
                      f'{estimate(metrics, f"{name}_{head}_forced_mate_wrong_advantage")} |')
    current = [r for r in rows if r['actor300_misorders']]
    report += ['', f'Actor300 still misorders {len(current)}/{len(rows)} paired initial moves. '
               f'{sum(r["endpoints"]["physical"] is not None for r in current)} of those have physical endpoints.',
               'Actor300 main physical-endpoint error on this subset: '
               + estimate(summary['actor_misordered_metrics'], 'actor300_main_physical_error') + ' points.', '',
               '## Physical endpoints, sorted by actor300 main error', '',
               '| Position | Original side | Ply | Reference | Start main | Actor main | Actor aux |',
               '|---|---|---:|---:|---:|---:|---:|']
    physical = sorted([r for r in rows if r['endpoints']['physical'] is not None],
                      key=lambda r: r['actor300_main_physical_error'], reverse=True)
    for r in physical:
        report.append(f'| {r["position_id"]} | {"White" if r["root_white"] else "Black"} | '
            f'{r["endpoints"]["physical"]} | ' + ' | '.join(percent(r[k]) for k in
            ['actor300_main_physical_reference', 'start_main_physical_prediction',
             'actor300_main_physical_prediction', 'actor300_aux_physical_prediction']) + ' |')
    report += ['', '## Verification and interpretation limits', '',
        '- Both checkpoints are frozen; no training, loss change, or replay mutation occurred.',
        '- Exact histories, FENs, legal alignment, colors, and cached-versus-full inference were checked.',
        '- Engine searches use independent fresh hashes at 100,000 and 1,000,000 nodes per position. '
        'The latter selects continuation moves. Mate scores are retained separately.',
        '- Physical endpoints require a material deficit/promotion, two following quiet plies, '
        'and stable strongly losing engine evaluations. This operational definition can miss other tactical consequences.',
        '- Terminal states use game rules and are excluded from neural-error metrics. '
        'Unresolved lines are not classified as failures.',
        '- Main and auxiliary heads share value features. Their agreement does not identify why an error exists.',
        '- This test can locate recognition failures; it cannot alone separate representation, supervision, '
        'optimization, or training-policy/evaluation-policy mismatch.',
        '- Starting auxiliary weights were untrained, so their predictions are retained in CSV but omitted from comparison tables.',
        '', 'Files: `positions.csv`, `continuations.pgn`, `summary.json`, `lines/`, checkpoint prediction directories, '
        '`verification-*.json`, `manifest.json`, and `PROTOCOL.md`.', '']
    report += ['## Completion and engine stability', '',
        f'Engine states: {len(flat)}; nonterminal neural states per checkpoint: '
        f'{sum(r["actor_main"] is not None for r in flat)}; terminal states: '
        f'{sum(bool(r["terminal"]) for r in flat)}.',
        'States with an expected-score change greater than 10 points between engine budgets: '
        f'{sum(r["engine_100k"] is not None and abs(r["engine_100k"] - r["engine_1m"]) > .10 for r in flat)}. '
        'These states are excluded from qualifying recognition endpoints.', '']
    rechecked = []
    for pos in manifest['positions']:
        states = {a: lines[pos['position_id'], a]['states'][1] for a in ('bad', 'good')}
        scores = {a: s['terminal']['expectation'] if s['terminal'] else s['verified']['expectation']
                  for a, s in states.items()}
        rechecked.append(dict(position_id=pos['position_id'], **scores,
                              good_minus_bad=scores['good'] - scores['bad']))
    report += ['Independent 1-million-node recheck immediately after the paired moves: '
        f'{sum(r["good_minus_bad"] > .02 for r in rechecked)}/{len(rechecked)} retain a good-minus-bad '
        'reference gap greater than 2 points. The full recheck is saved in `paired-reference-recheck.json`.', '']
    atomic_json(directory / 'paired-reference-recheck.json', rechecked)
    excluded = Counter()
    for pos in manifest['positions']:
        row = next(r for r in rows if r['position_id'] == pos['position_id'])
        if row['endpoints']['physical'] is not None:
            continue
        states = lines[pos['position_id'], 'bad']['states']
        candidates = []
        for i, s in enumerate(states):
            tail = states[i + 1:i + 3]
            if s['ply'] < 1 or s['terminal'] or s['in_check'] or len(tail) != 2:
                continue
            if any(t['capture'] or t['promotion'] or t['in_check'] or t['terminal'] for t in tail):
                continue
            deficit = all(t['material'] <= pos['initial_material'] - 3 for t in [s] + tail)
            promotion = any(t['promotion'] and t['mover_white'] != pos['root_white'] for t in states[:i + 1])
            if deficit or promotion:
                candidates.append(s)
        if not candidates:
            excluded['no_settled_material_or_promotion_within_horizon'] += 1
        elif not any(s.get('stable') for s in candidates):
            excluded['physical_event_but_engine_budget_disagreement'] += 1
        else:
            excluded['physical_event_but_not_stable_strongly_losing'] += 1
    report += ['Reasons for exclusion from the strict physical-endpoint comparison: '
               + '; '.join(f'{k}: {v}' for k, v in excluded.items()) + '.', '']
    (directory / 'REPORT.md').write_text('\n'.join(report))

    # Standalone vector charts with hoverable observations; no external assets or dependencies.
    pages = ['<!doctype html><meta charset="utf-8"><title>Tactical recognition</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;color:#18212b} '
        'svg{width:100%;max-width:1000px;background:#fafafa} section{margin:35px 0} '
        'p{line-height:1.5}</style>', '<h1>Frozen tactical recognition</h1>',
        '<p>All scores are for the original player. Blue: Stockfish; red: actor300 main; '
        'orange: actor300 auxiliary; gray: starting main. Green vertical line: first physical endpoint. '
        'Hover over an observation for its value. Cases with physical endpoints come first, sorted by actor error.</p>']
    ordered = physical + [r for r in rows if r['endpoints']['physical'] is None]
    colors = dict(engine='#1767b2', actor_main='#cb3039', actor_aux='#c17b00', start_main='#6c7580')
    for row in ordered:
        pid = row['position_id']
        line, pred = lines[pid, 'bad'], predictions[pid, 'bad']
        pages += [f'<section><h2>{pid}</h2><p>{html.escape(row["source"])}; '
                  f'physical endpoint: {row["endpoints"]["physical"]}; '
                  f'forced-mate endpoint: {row["endpoints"]["forced_mate"]}</p>',
                  '<svg viewBox="0 0 1000 300" role="img" aria-label="Expected score by continuation ply">']
        def xy(i, value): return 55 + i * 52, 255 - 220 * value
        for v in (0, .25, .5, .75, 1):
            y = xy(0, v)[1]
            pages.append(f'<line x1="55" x2="945" y1="{y}" y2="{y}" stroke="#ddd"/>'
                         f'<text x="8" y="{y+5}">{v*100:.0f}%</text>')
        for i in range(18):
            x = xy(i, 0)[0]
            pages.append(f'<text x="{x-5}" y="278">{i}</text>')
        end = row['endpoints']['physical']
        if end is not None:
            x = xy(end, 0)[0]
            pages.append(f'<line x1="{x}" x2="{x}" y1="30" y2="255" stroke="#208740" stroke-dasharray="5 4"/>')
        for series, color in colors.items():
            points = []
            for s in line['states']:
                i = s['ply']
                if s['terminal']: continue
                value = (s['verified']['expectation'] if series == 'engine' else
                         pred['actor300' if series.startswith('actor') else 'start'][i][series.split('_')[1]])
                x, y = xy(i, value)
                points.append(f'{x},{y}')
                pages.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"><title>'
                             f'{series}, ply {i}: {percent(value)}</title></circle>')
            pages.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        pages.append('</svg></section>')
    (directory / 'trajectories.html').write_text('\n'.join(pages))
    atomic_json(directory / 'report-provenance.json', dict(script_sha256=file_hash(Path(__file__)),
                summary_sha256=file_hash(directory / 'summary.json'),
                state_rows=len(flat), neural_rows=sum(r['actor_main'] is not None for r in flat)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    render(parser.parse_args().directory)
