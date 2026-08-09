import subprocess

import pytest

import telemetry_metrics


def test_read_uptime_seconds(tmp_path):
    path = tmp_path / "uptime"
    path.write_text("123.45 999.00\n", encoding="utf-8")

    assert telemetry_metrics.read_uptime_seconds(path) == 123.45


def test_read_cpu_temperature_celsius(tmp_path):
    path = tmp_path / "temp"
    path.write_text("68400\n", encoding="utf-8")

    assert telemetry_metrics.read_cpu_temperature_celsius(path) == 68.4


def test_read_memory_used_ratio(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemTotal: 1000 kB\nMemAvailable: 250 kB\n", encoding="utf-8")

    assert telemetry_metrics.read_memory_used_ratio(path) == 0.75


def test_read_cpu_throttled_parses_vcgencmd_flags():
    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, stdout="throttled=0x50000\n", stderr="")

    assert telemetry_metrics.read_cpu_throttled(run=runner) == 1


def test_read_systemd_unit_returns_status_and_restart_count():
    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [],
            0,
            stdout="NRestarts=3\nActiveState=active\n",
            stderr="",
        )

    assert telemetry_metrics.read_systemd_unit("streamer.service", run=runner) == {
        "unit": "streamer.service",
        "up": 1,
        "restart_count": 3,
    }


def test_collect_status_snapshot_contains_service_records(monkeypatch):
    monkeypatch.setattr(telemetry_metrics, "read_uptime_seconds", lambda: 10.0)
    monkeypatch.setattr(telemetry_metrics, "read_cpu_temperature_celsius", lambda: 60.0)
    monkeypatch.setattr(telemetry_metrics, "read_cpu_throttled", lambda: 0)
    monkeypatch.setattr(telemetry_metrics, "read_memory_used_ratio", lambda: 0.25)
    monkeypatch.setattr(telemetry_metrics, "read_disk_used_ratio", lambda: 0.5)
    monkeypatch.setattr(telemetry_metrics, "load_bitrate_bps", lambda: 15_000_000)
    monkeypatch.setattr(
        telemetry_metrics,
        "read_systemd_unit",
        lambda unit: {"unit": unit, "up": 1, "restart_count": 0},
    )
    monkeypatch.setattr(
        telemetry_metrics,
        "HAILO_DEVICE_PATH",
        type("FakePath", (), {"exists": lambda self: True})(),
    )

    snapshot = telemetry_metrics.collect_status_snapshot()

    assert snapshot["device_up"] == 1
    assert snapshot["hailo_present"] == 1
    assert snapshot["stream_bitrate_bits_per_second"] == 15_000_000
    assert [service["unit"] for service in snapshot["services"]] == list(
        telemetry_metrics.SYSTEMD_UNITS
    )
