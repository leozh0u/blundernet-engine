from blundernet import data


def pgn(game_id, day=1):
    return (
        f'[GameId "{game_id}"]\n[UTCDate "2026.10.{day:02}"]\n'
        '[UTCTime "12:00:00"]\n[Result "1-0"]\n\n'
        '1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 1-0\n\n'
    )


def test_incremental_fetch_uses_oldest_first(monkeypatch):
    class Response:
        status_code = 200
        text = ""

        def raise_for_status(self):
            pass

    calls = []
    monkeypatch.setattr(data.requests, "get", lambda *a, **kw: calls.append(kw) or Response())
    data.fetch_games("player", 123, max_games=2)
    assert calls[0]["params"]["sort"] == "dateAsc"


def test_cursor_advances_only_to_the_last_exported_game(monkeypatch):
    state = {"cursor": {"player": 1}, "seen_ids": [], "rotation": 0,
             "total_games": 0, "total_positions": 0}
    monkeypatch.setattr(data, "load_state", lambda: state)
    monkeypatch.setattr(data, "save_state", lambda s: None)
    monkeypatch.setattr(data, "get_players", lambda: ["player"])
    monkeypatch.setattr(data, "fetch_games", lambda *args: pgn("old", 1) + pgn("new", 2))
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    data.gather_batch(n_players=1, max_players=1)
    assert state["cursor"]["player"] == 1790942400000  # 2026-10-02 12:00 UTC


def test_seen_ids_are_not_evicted_lexicographically(monkeypatch):
    ids = [f"z{i:07}" for i in range(20000)]
    state = {"cursor": {}, "seen_ids": ids, "rotation": 0,
             "total_games": 20000, "total_positions": 0}
    monkeypatch.setattr(data, "load_state", lambda: state)
    monkeypatch.setattr(data, "save_state", lambda s: None)
    monkeypatch.setattr(data, "get_players", lambda: ["player"])
    monkeypatch.setattr(data, "fetch_games", lambda *args: pgn("a-new-id"))
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    data.gather_batch(n_players=1, max_players=1)
    assert "a-new-id" in state["seen_ids"]
    assert len(state["seen_ids"]) == 20001
