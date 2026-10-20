from bench.load.outputs import LoadResult


class TestLoadResult:
    def test_one_line_summary(self) -> None:
        result = LoadResult(requests=3000, errors=0, rps=302.8, p50_ms=42, p99_ms=140)

        assert repr(result) == "302.8 req/s  p50 42.0 ms  p99 140.0 ms  errors 0"
