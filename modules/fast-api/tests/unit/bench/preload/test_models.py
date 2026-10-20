from bench.preload.models import Metrics


class TestMetrics:
    def test_median_picks_a_real_run(self) -> None:
        runs = [Metrics.model_validate({field: value for field in Metrics.model_fields}) for value in (1, 2, 3, 4)]

        assert Metrics.median(runs).errors == 2
