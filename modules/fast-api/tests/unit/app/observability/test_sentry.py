import pytest
import sentry_sdk
from pydantic import ValidationError
from pytest_mock import MockerFixture

from app.common.enums.environment import Environment
from app.core.config import ModelSettings
from app.core.config import SentrySettings
from app.core.preload import preload
from app.inference.runtime.served import ServedModel
from app.observability.sentry import start_sentry
from app.observability.sentry import tag_request_id


class TestSentrySettings:
    def test_enabled_needs_a_dsn(self) -> None:
        with pytest.raises(ValidationError, match="SENTRY_ENABLED needs SENTRY_DSN"):
            SentrySettings(enabled=True)

    def test_reads_the_sentry_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTRY_ENABLED", "true")
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.example/1")

        assert SentrySettings().dsn == "https://key@sentry.example/1"


class TestStartSentry:
    def test_does_nothing_when_disabled(self, mocker: MockerFixture) -> None:
        init = mocker.patch.object(sentry_sdk, "init")
        start_sentry(SentrySettings(), Environment.PRODUCTION)
        init.assert_not_called()

    def test_reports_errors_only_without_personal_data(self, mocker: MockerFixture) -> None:
        init = mocker.patch.object(sentry_sdk, "init")
        settings = SentrySettings(enabled=True, dsn="https://key@sentry.example/1", release="1.2.3")

        start_sentry(settings, Environment.PRODUCTION)

        init.assert_called_once_with(
            dsn="https://key@sentry.example/1",
            environment="production",
            release="1.2.3",
            traces_sample_rate=0.0,
            send_default_pii=False,
            before_send=tag_request_id,
        )


class TestPreload:
    def test_a_model_that_fails_to_load_is_reported_then_raised(
        self, tiny_model_settings: ModelSettings, mocker: MockerFixture
    ) -> None:
        failure = RuntimeError("corrupt model")
        mocker.patch.object(ServedModel, ServedModel.load.__name__, side_effect=failure)
        report = mocker.patch("app.core.preload.report")

        with pytest.raises(RuntimeError, match="corrupt model"):
            preload(tiny_model_settings)

        report.assert_called_once_with(failure)
