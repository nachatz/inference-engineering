import psutil


def children(pid: int) -> list[int]:
    return sorted(child.pid for child in psutil.Process(pid).children())


def oldest_gunicorn() -> int:
    masters = [
        process for process in psutil.process_iter(["name", "create_time"]) if process.info["name"] == "gunicorn"
    ]
    if not masters:
        raise LookupError("no gunicorn process is running; start one with `just serve` or set MEM_PID")

    return min(masters, key=lambda process: process.info["create_time"]).pid


def cpu_seconds(pid: int) -> float:
    times = psutil.Process(pid).cpu_times()
    return times.user + times.system


def listening_sockets(port: int) -> int:
    """How many sockets listen on `port`: one per worker with reuse_port, else gunicorn's one."""
    return sum(
        1
        for connection in psutil.net_connections(kind="tcp")
        if connection.status == psutil.CONN_LISTEN and connection.laddr and connection.laddr.port == port
    )
