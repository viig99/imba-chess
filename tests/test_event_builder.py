from imba_chess.data.event_builder import BOS_SEQ_TOKEN_ID, EventBuilder
from imba_chess.data.lichess_dataset import LichessDataset
from imba_chess.data.move_vocab import MoveVocab


def _row():
    return {
        "Event": "Rated Blitz game",
        "Site": "https://lichess.org/example",
        "UTCDate": "2026-01-01",
        "UTCTime": "12:00:00",
        "White": "Alice",
        "Black": "Bob",
        "WhiteElo": "2200",
        "BlackElo": "2200",
        "Result": "1-0",
        "TimeControl": "300+0",
        "Termination": "Normal",
        "ECO": "C20",
        "Opening": "King's Pawn Game",
        "movetext": "1. e4 e5 2. Nf3 Nc6 1-0",
    }


def test_event_builder_builds_bos_plus_plies():
    dataset = LichessDataset(min_avg_elo=2000)
    game = list(dataset.stream_from_rows([_row()]))[0]
    vocab = MoveVocab.build_from_games([game])

    builder = EventBuilder(vocab)
    sample = builder.build_game(game)

    # 4 plies + BOS
    assert len(sample["seq_token_id"]) == 5
    assert sample["seq_token_id"][0] == BOS_SEQ_TOKEN_ID
    assert sample["loss_mask"][0] == 0
    assert sample["loss_mask"][1:] == [1, 1, 1, 1]
    assert sample["prev_move_id"][1] == vocab.start_id
    assert len(sample["piece_ids"][1]) == 64

