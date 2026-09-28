"""Render primary frozen linear probes and the separate exploratory branch refit."""
import argparse
import json
from pathlib import Path

import numpy as np

from imba_chess.data.self_play_store import atomic_json
from imba_chess.self_play.seeds import file_hash
from scripts.audit_frozen_value_probes import grouped_summary


def main(directory):
    primary = json.loads((directory / 'summary.json').read_text())
    branch = json.loads((directory / 'branch_refit/summary.json').read_text())
    metrics = dict(primary['metrics'], value_branch=branch['metrics']['value_branch'])
    physical = primary['physical_cases'] + [r for r in branch['physical_cases'] if r['probe'] == 'value_branch']
    names = dict(baseline='Original actor300', training_prior='Constant training-prior control',
                 output_only='Output-only recalibration (12 parameters)', value512='512-feature linear probe',
                 shared1024='1,024-feature linear probe', value_branch='Private value branch refit')
    rows = json.loads((directory / 'rows.json').read_text())
    oof = json.loads((directory / 'out_of_fold.json').read_text())['probabilities']
    oof['value_branch'] = json.loads((directory / 'branch_refit/out_of_fold.json').read_text())['probabilities']['value_branch']
    pairs = {}
    for i, row in enumerate(rows):
        if row['ply'] == 1:
            pairs.setdefault(row['position_id'], {})[row['arm']] = i
    pairs = [p for _,p in sorted(pairs.items()) if set(p) == {'good', 'bad'}]
    pair_rows = [rows[p['bad']] for p in pairs]
    correct = {}
    for name, probabilities in oof.items():
        p = np.array(probabilities)
        exp = p[:, 2] + .5 * p[:, 1]
        exp = np.where([r['root_white'] != r['turn_white'] for r in rows], 1 - exp, exp)
        correct[name] = np.array([float(exp[p['good']] > exp[p['bad']] + 1e-12) for p in pairs])
    paired = {name: grouped_summary(v - correct['baseline'], pair_rows) for name,v in correct.items()}
    atomic_json(directory / 'paired-ordering-contrasts.json', paired)
    report = ['# Frozen value probes: findings', '',
        'The cheap probes did not clearly improve held-out expected-score accuracy or tactical move ordering. '
        'WDL log loss improved substantially, but almost the same gain came from remapping the three existing outputs. '
        'That is consistent with reducing overconfidence against the engine targets, without demonstrating a tactical fix.', '',
        'All predictions below are held out by original source game. Five folds cover 42 games, 45 audited '
        'move pairs and 1,588 nonterminal observations. Both branches and all neighboring positions from '
        'a game remain together. Hyperparameters were selected using separate inner-validation games.', '',
        '| Readout | WDL cross entropy (nats, lower better) | Expected-score MAE (points) | Good move preferred |',
        '|---|---:|---:|---:|']
    for name,title in names.items():
        m = metrics[name]
        report.append(f'| {title} | {m["all"]["ce"]["mean"]:.3f} | '
                      f'{100*m["all"]["mae"]["mean"]:.2f} | {m["pair_ordering"]["correct_pairs"]}/45 |')
    report += ['', 'The constant control ties every pair; its 0/45 is not a 0% decisive ranking accuracy. '
        'Even that control improves log loss, which shows why log-loss improvement alone is insufficient here.', '',
        '## Paired expected-score error changes', '',
        'Probe minus original, percentage points, with 95% source-game bootstrap intervals. Negative is better.', '',
        '| Probe | Change | Interval |', '|---|---:|---:|']
    for name in ('output_only', 'value512', 'shared1024', 'value_branch'):
        d = metrics[name]['all']['mae_minus_baseline']
        report.append(f'| {names[name]} | {100*d["mean"]:+.2f} | '
                      f'[{100*d["ci95"][0]:+.2f}, {100*d["ci95"][1]:+.2f}] |')
    report += ['', 'Every interval includes zero. The stable-engine subset also shows no clear expected-score '
               'improvement. Two additional correct pairs from the shared-feature probe do not establish a ranking gain.', '',
               '## The three previously identified physical-loss cases', '',
               'Expected score for the original, losing player. Each case is predicted by a model fitted without '
               'any positions from its source game. Engine references are 0%, 0.15%, and 0%.', '',
               '| Readout | Rook for pawn | Queen-loss sequence | Knight captured |',
               '|---|---:|---:|---:|']
    pids = ['069af062fab09adb', '306b913efaecdca0', '6087dfeaf838ad5d']
    for name in ('baseline', 'output_only', 'value512', 'shared1024', 'value_branch'):
        values = {r['position_id']: r['prediction'] for r in physical if r['probe'] == name}
        report.append('| ' + names[name] + ' | ' + ' | '.join(f'{100*values[pid]:.1f}%' for pid in pids) + ' |')
    report += ['', 'Some predictions become less optimistic, but none of the tested probes brings these cases '
        'within the prior 15-point recognition tolerance. A shift toward 50% is not by itself recognition '
        'of a strongly losing position.', '',
        '## Interpretation', '',
        '- A new final linear output was not a clear fix on unseen games in this sample.',
        '- A linear readout directly from the shared features was not a clear fix either.',
        '- Refitting all 2,629,635 parameters of the existing private value branch while keeping the shared '
        'model frozen did not clearly improve those tactical metrics under this small fitting protocol.',
        '- These negative results do not prove the shared representation lacks the information. Only 42 '
        'independent games are available, probes have limited training/tuning budgets, and this set was '
        'selected from historical blunders. Broader data or a different readout might generalize differently.',
        '- The linear fits used strongly regularized, validation-selected solutions. This is a held-out '
        'generalization test, not a proof that the features cannot memorize corrected examples.',
        '- The private-branch refit was an exploratory follow-up to the linear results. It used the same '
        'game splits and inner validation, but its decision to run followed inspection of the primary results.',
        '- The original model, full input histories, self-play loss, and replay were unchanged. Only new '
        'diagnostic readouts/copies were fitted. No checkpoint is promoted.',
        '- History truncation was not part of this experiment. These results neither establish nor exclude '
        'harmful dependence on older history. A separate fixed-weight input ablation would test that.', '',
        '## Cost and checks', '',
        'Feature extraction took 5.8 seconds for 1,543 unique histories. The linear fitting command took '
        '6.7 seconds, and the private-branch command took 13.0 seconds, excluding implementation and review.',
        'Five unit tests covered target perspective/order, disjoint nested source-game folds, game weighting, '
        'recovery of a known held-out synthetic signal, and clustered summary arithmetic.',
        'Original outputs reconstructed from the cached features agree within 9e-7 WDL probability. '
        'Checkpoint and cache hashes are verified separately in delivery-verification.json.', '',
        'Artifacts: manifest.json, PROTOCOL.md, feature-verification.json, rows.json, features.pt, '
        'folds/, probe_weights/, out_of_fold.json, summary.json, paired-ordering-contrasts.json, '
        'and branch_refit/.', '']
    (directory / 'REPORT.md').write_text('\n'.join(report))
    atomic_json(directory / 'report-provenance.json', dict(script_sha256=file_hash(Path(__file__)),
        primary_summary_sha256=file_hash(directory / 'summary.json'),
        branch_summary_sha256=file_hash(directory / 'branch_refit/summary.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    main(parser.parse_args().directory)
