import asyncio
import threading

from ytds.events import EventBus


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _progress(ds="d1", vid="v1", stage="download", done=1, total=10):
    return {"type": "progress", "dataset_id": ds, "video_id": vid, "stage": stage, "done": done, "total": total}


async def _drain(q):
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def run(coro):
    return asyncio.run(coro)


def test_progress_throttled_to_4_per_second():
    async def go():
        clock = FakeClock()
        bus = EventBus(clock=clock)
        bus.bind_loop(asyncio.get_running_loop())
        q = bus.subscribe("d1")
        for i in range(20):  # 20 events over 1 second at 50 ms spacing
            bus.publish(_progress(done=i + 1, total=100))
            clock.t += 0.05
        return await _drain(q)

    assert len(run(go())) == 4


def test_final_progress_always_sent():
    async def go():
        bus = EventBus(clock=FakeClock())
        bus.bind_loop(asyncio.get_running_loop())
        q = bus.subscribe("d1")
        bus.publish(_progress(done=1, total=5))
        bus.publish(_progress(done=4, total=5))  # throttled
        bus.publish(_progress(done=5, total=5))  # final, kept
        return await _drain(q)

    assert [e["done"] for e in run(go())] == [1, 5]


def test_throttle_is_per_video_and_stage():
    async def go():
        bus = EventBus(clock=FakeClock())
        bus.bind_loop(asyncio.get_running_loop())
        q = bus.subscribe("d1")
        bus.publish(_progress(vid="a"))
        bus.publish(_progress(vid="b"))
        bus.publish(_progress(vid="a", stage="segment"))
        return await _drain(q)

    assert len(run(go())) == 3


def test_stage_events_never_throttled():
    async def go():
        bus = EventBus(clock=FakeClock())
        bus.bind_loop(asyncio.get_running_loop())
        q = bus.subscribe("d1")
        types = ["stage_done", "stage_failed", "video_done", "log", "llm_status"]
        for t in types:
            for _ in range(3):
                bus.publish({"type": t, "dataset_id": "d1", "video_id": "v1", "stage": "download"})
        return await _drain(q)

    assert len(run(go())) == 15


def test_subscribers_only_get_their_dataset():
    async def go():
        bus = EventBus(clock=FakeClock())
        bus.bind_loop(asyncio.get_running_loop())
        q1, q2 = bus.subscribe("d1"), bus.subscribe("d2")
        bus.publish({"type": "log", "dataset_id": "d1", "msg": "x"})
        got = (await _drain(q1), await _drain(q2))
        bus.unsubscribe("d1", q1)
        bus.publish({"type": "log", "dataset_id": "d1", "msg": "y"})
        await asyncio.sleep(0)
        return got, q1.empty()

    (g1, g2), q1_empty = run(go())
    assert len(g1) == 1 and g2 == [] and q1_empty


def test_publish_before_bind_loop_is_dropped_without_error():
    bus = EventBus(clock=FakeClock())
    bus.publish({"type": "log", "dataset_id": "d1"})  # must not raise


def test_publish_from_worker_thread():
    async def go():
        bus = EventBus(clock=FakeClock())
        bus.bind_loop(asyncio.get_running_loop())
        q = bus.subscribe("d1")
        t = threading.Thread(target=bus.publish, args=({"type": "log", "dataset_id": "d1"},))
        t.start()
        t.join()
        return await asyncio.wait_for(q.get(), 1)

    assert run(go())["type"] == "log"
