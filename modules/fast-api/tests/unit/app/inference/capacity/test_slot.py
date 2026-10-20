import os
from unittest.mock import Mock

import anyio
import pytest
from prometheus_client import REGISTRY
from pytest_mock import MockerFixture

from app.inference.capacity.slot import InferenceSlot
from app.inference.capacity.slot import exit_worker
from app.inference.capacity.slot import watch
from app.inference.exceptions import Overloaded


def sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


async def hold(slot: InferenceSlot, entered: anyio.Event, release: anyio.Event) -> None:
    async with slot.acquire():
        entered.set()
        await release.wait()


async def refused(slot: InferenceSlot, reason: str) -> None:
    with pytest.raises(Overloaded, match=reason):
        async with slot.acquire():
            pass


class TestInferenceSlot:
    async def test_is_busy_only_while_held(self) -> None:
        slot = InferenceSlot(max_queue=1, max_queue_wait_s=1.0)
        async with slot.acquire():
            await anyio.sleep(0.01)
            assert slot.busy_for() > 0
            assert sample("predict_inflight") == 1
        assert slot.busy_for() == 0
        assert sample("predict_inflight") == 0

    async def test_queues_up_to_max_queue_then_refuses(self) -> None:
        slot = InferenceSlot(max_queue=1, max_queue_wait_s=5.0)
        entered, release = anyio.Event(), anyio.Event()
        shed = sample("predict_shed_total", reason="queue_full")

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(hold, slot, entered, release)
            await entered.wait()
            tasks.start_soon(hold, slot, anyio.Event(), release)  # waits behind the first
            await anyio.sleep(0.01)

            assert (slot.waiting, sample("predict_queue_depth")) == (1, 1)
            await refused(slot, "queue is full")
            release.set()

        assert sample("predict_shed_total", reason="queue_full") == shed + 1
        assert sample("predict_queue_depth") == 0

    async def test_refuses_a_request_that_waited_past_the_deadline(self) -> None:
        slot = InferenceSlot(max_queue=1, max_queue_wait_s=0.01)
        entered, release = anyio.Event(), anyio.Event()
        shed = sample("predict_shed_total", reason="queue_timeout")

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(hold, slot, entered, release)
            await entered.wait()
            tasks.start_soon(refused, slot, "waited too long")
            await anyio.sleep(0.05)

            # Refused at the deadline, while the first request still holds the slot.
            assert sample("predict_shed_total", reason="queue_timeout") == shed + 1
            assert sample("predict_queue_depth") == 0
            release.set()

        assert slot.busy_for() == 0


class TestWatch:
    async def test_reports_a_stuck_inference_once(self) -> None:
        stuck = Mock(spec=InferenceSlot)
        stuck.busy_for.return_value = 10.0
        on_stuck = Mock()
        with anyio.fail_after(5):
            await watch(stuck, stuck_after_s=0.04, on_stuck=on_stuck)
        on_stuck.assert_called_once_with(10.0)

    async def test_leaves_a_working_slot_alone(self) -> None:
        on_stuck = Mock()
        with anyio.move_on_after(0.1):
            await watch(InferenceSlot(max_queue=1, max_queue_wait_s=1.0), stuck_after_s=0.04, on_stuck=on_stuck)
        on_stuck.assert_not_called()


class TestExitWorker:
    def test_logs_then_exits_so_gunicorn_replaces_the_worker(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        exit_ = mocker.patch.object(os, os._exit.__name__)
        exit_worker(31.24)
        exit_.assert_called_once_with(1)
        assert "inference has run for 31.2 s" in caplog.text
