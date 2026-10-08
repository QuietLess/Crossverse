import requests

from crossverse.data import amazon

DATA = bytes(range(256)) * 40


class _FakeResponse:
    def __init__(self, start: int, drop_after: int | None):
        self.status_code = 206 if start else 200
        self.headers = {"Content-Length": str(len(DATA) - start)}
        self._body, self._drop_after = DATA[start:], drop_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), 1000):
            if self._drop_after is not None and i >= self._drop_after:
                raise requests.ConnectionError("Read timed out.")
            yield self._body[i:i + 1000]


def test_download_resumes_after_dropped_connection(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, stream, timeout, headers):
        start = int(headers["Range"][6:-1]) if headers else 0
        calls.append(start)
        return _FakeResponse(start, drop_after=3000 if len(calls) == 1 else None)

    monkeypatch.setattr(amazon.requests, "get", fake_get)
    monkeypatch.setattr(amazon.time, "sleep", lambda s: None)
    tmp = tmp_path / "file.part"
    amazon._download_resumable("https://example/file", tmp)
    assert tmp.read_bytes() == DATA
    assert calls == [0, 3000]  # second request continued where the first one stopped


def test_many_stalls_with_progress_still_complete(tmp_path, monkeypatch):
    def fake_get(url, stream, timeout, headers):
        start = int(headers["Range"][6:-1]) if headers else 0
        return _FakeResponse(start, drop_after=1000)  # every connection dies after 1 KB

    monkeypatch.setattr(amazon.requests, "get", fake_get)
    monkeypatch.setattr(amazon.time, "sleep", lambda s: None)
    tmp = tmp_path / "file.part"
    amazon._download_resumable("https://example/file", tmp, attempts=3)  # 11 stalls, but each makes progress
    assert tmp.read_bytes() == DATA
