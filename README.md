# pi_fsw
repo for sw running on pi

## Device telemetry

The camera exposes a local JSON status snapshot at `GET /telemetry` on port 5001.
The Qwatercam workstation application polls this endpoint and relays the metrics to
Grafana. The camera itself does not need internet access or Grafana credentials.

Example device metadata:

```json
{
  "serial_number": "0002"
}
```

Example request:

```bash
curl http://192.168.4.2:5001/telemetry
```

Every payload includes the validated device serial number, firmware version,
observation time, scalar health values, and systemd service state. The serial
number in `/opt/device_metadata.json` is the source of truth.
