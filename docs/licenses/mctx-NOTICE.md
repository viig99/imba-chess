The sequential-halving visit schedule in `src/imba_chess/eval/gumbel_search.py`
is adapted from Google DeepMind mctx, revision
`88f92056a420c2673bed282f5a0c00211f126e78`.

Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
Licensed under the Apache License, Version 2.0; see `mctx-LICENSE`.

The completed-Q and action-selection rules also use this revision as their
algorithm reference. Independent fixtures were generated with that reference
implementation. JAX and mctx are not runtime dependencies of imba-chess.
