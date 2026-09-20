import base64
import csv
import os
import queue
import signal
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update
import requests


# SETTINGS

CAPTURE_SECONDS = 10
DASH_INTERVAL_MS = 250
MAX_VISIBLE_POINTS = 300
DISPLAY_INTERVAL_SECONDS = 0.05  # 20 Hz on the live graph only; raw CSV stays full-rate
CAMERA_INDEX = 0
OUTPUT_DIR = Path("captures_6D")
SESSION_LOG = OUTPUT_DIR / "segment_manifest.csv"
PHYPHOX_URL = os.getenv("PHYPHOX_URL", "http://192.168.10.144:8080").rstrip("/")
PHONE_POLL_SECONDS = 0.05

AXES = ("x", "y", "z")


# SHARED STATE 

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

latest = {"x": None, "y": None, "z": None}
fresh_axes = set()
state_lock = threading.Lock()

# live events are used only for the smooth browser graph.
live_queue = queue.Queue(maxsize=5000)

# complete xyz rows are used for the 10-second files.
segment_rows = []
segment_lock = threading.Lock()

latest_segment = {
    "version": 0,
    "basename": None,
    "df": None,
    "image_src": None,
    "status": "Waiting for first 10-second segment...",
}
latest_segment_lock = threading.Lock()

last_callback_time = {"x": None, "y": None, "z": None}
callback_lock = threading.Lock()

stop_event = threading.Event()


# HELPERS

def next_sequence_number() -> int:
    seqs = []
    for p in OUTPUT_DIR.glob("*.csv"):
        if p.name == SESSION_LOG.name:
            continue
        m = re.match(r"^(\d+)_\d{14}\.csv$", p.name)
        if m:
            seqs.append(int(m.group(1)))
    return max(seqs, default=0) + 1


def safe_put_live(event):
    try:
        live_queue.put_nowait(event)
    except queue.Full:
        try:
            live_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            live_queue.put_nowait(event)
        except queue.Full:
            pass


def append_manifest(basename, csv_path, jpg_path, row_count, note=""):
    new_file = not SESSION_LOG.exists()
    with SESSION_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                ["basename", "csv_file", "jpg_file", "rows", "saved_at", "note"]
            )
        writer.writerow(
            [
                basename,
                csv_path.name,
                jpg_path.name if jpg_path else "",
                row_count,
                datetime.now().isoformat(timespec="seconds"),
                note,
            ]
        )


def frame_to_data_uri(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    encoded = base64.b64encode(buf.tobytes()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def make_segment_figure(df: pd.DataFrame, title: str):
    fig = go.Figure()
    for axis in AXES:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[axis],
                mode="lines",
                name=axis.upper(),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Acceleration",
        margin=dict(l=45, r=20, t=50, b=45),
        uirevision="segment",
    )
    return fig


def empty_live_figure():
    fig = go.Figure()
    for axis in AXES:
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name=axis.upper()))
    fig.update_layout(
        title="Live smartphone accelerometer",
        xaxis_title="Time received by Python",
        yaxis_title="Acceleration",
        margin=dict(l=45, r=20, t=50, b=45),
        uirevision="live",
    )
    return fig


# DATA CAPTURE 

def handle_axis_update(axis, value, sample_datetime=None, push_live=True):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return

    now = sample_datetime or datetime.now()
    timestamp = now.isoformat(timespec="milliseconds")

    with callback_lock:
        last_callback_time[axis] = datetime.now()

    if push_live:
        safe_put_live((axis, timestamp, numeric_value))

    complete_row = None
    with state_lock:
        latest[axis] = numeric_value
        fresh_axes.add(axis)

        if all(a in fresh_axes for a in AXES):
            complete_row = (
                timestamp,
                latest["x"],
                latest["y"],
                latest["z"],
            )
            fresh_axes.clear()

    if complete_row is not None:
        with segment_lock:
            segment_rows.append(complete_row)


def _phy_values(payload, name):
    item = payload.get("buffer", {}).get(name, {})
    vals = item.get("buffer", []) if isinstance(item, dict) else item
    if not isinstance(vals, list):
        return []
    result = []
    for v in vals:
        try:
            if v is not None:
                result.append(float(v))
        except (TypeError, ValueError):
            pass
    return result


def discover_phyphox_buffers():
    r = requests.get(f"{PHYPHOX_URL}/config", timeout=3)
    r.raise_for_status()
    cfg = r.json()

    names = []
    for item in cfg.get("buffers", []):
        if isinstance(item, dict):
            name = item.get("name")
            if name:
                names.append(name)
        elif isinstance(item, str):
            names.append(item)

    print("phyphox buffers:", ", ".join(names))
    low = {n.lower(): n for n in names}

    def pick(*candidates):
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

    # common names used by phyphox built-in acceleration experiments
    t = pick("acc_time", "time", "lin_time")
    x = pick("accX", "x", "linX")
    y = pick("accY", "y", "linY")
    z = pick("accZ", "z", "linZ")

    if not all((t, x, y, z)):
        raise RuntimeError(
            "Could not identify acceleration buffers automatically. "
            f"Available buffers: {names}"
        )
    return t, x, y, z


def phone_worker():
    try:
        t_name, x_name, y_name, z_name = discover_phyphox_buffers()
        print(f"Connected directly to phone at {PHYPHOX_URL}")
        print(f"Using buffers: {t_name}, {x_name}, {y_name}, {z_name}")
    except Exception as exc:
        print(f"PHONE CONNECTION ERROR: {exc}")
        stop_event.set()
        return

    last_t = -1.0
    first_phone_t = None
    first_wall_time = None
    last_display_t = -1.0

    while not stop_event.is_set():
        try:
            # ask only for samples newer than last_t. x/y/z use time as their
            # reference buffer so the returned arrays remain aligned.
            params = {
                t_name: str(last_t),
                x_name: f"{last_t}|{t_name}",
                y_name: f"{last_t}|{t_name}",
                z_name: f"{last_t}|{t_name}",
            }
            r = requests.get(f"{PHYPHOX_URL}/get", params=params, timeout=2)
            r.raise_for_status()
            data = r.json()

            ts = _phy_values(data, t_name)
            xs = _phy_values(data, x_name)
            ys = _phy_values(data, y_name)
            zs = _phy_values(data, z_name)
            n = min(len(ts), len(xs), len(ys), len(zs))

            for i in range(n):
                phone_t = ts[i]

                # anchor phyphox elapsed time to the computer wall clock once,
                # then preserve the true spacing between sensor samples.
                if first_phone_t is None:
                    first_phone_t = phone_t
                    first_wall_time = datetime.now()

                sample_dt = first_wall_time + timedelta(seconds=(phone_t - first_phone_t))

                # keep the saved CSV at the phone's full sample rate.
                # thin only the live Dash graph to ~20 Hz so it stays readable.
                push_live = (
                    last_display_t < 0
                    or (phone_t - last_display_t) >= DISPLAY_INTERVAL_SECONDS
                )

                handle_axis_update("x", xs[i], sample_dt, push_live)
                handle_axis_update("y", ys[i], sample_dt, push_live)
                handle_axis_update("z", zs[i], sample_dt, push_live)

                if push_live:
                    last_display_t = phone_t

            if n:
                last_t = ts[n - 1]

        except requests.RequestException as exc:
            print(f"Phone connection warning: {exc}")
            time.sleep(0.5)
        except Exception as exc:
            print(f"Phone data warning: {exc}")
            time.sleep(0.5)

        time.sleep(PHONE_POLL_SECONDS)


def capture_worker():
    sequence = next_sequence_number()
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        with latest_segment_lock:
            latest_segment["status"] = (
                "ERROR: webcam could not be opened. Fix CAMERA_INDEX / permissions "
                "before collecting assessment data."
            )
        print(latest_segment["status"])
        camera = None
    else:
        # Let exposure settle.
        for _ in range(8):
            camera.read()

    next_deadline = time.monotonic() + CAPTURE_SECONDS

    try:
        while not stop_event.is_set():
            wait_time = max(0, next_deadline - time.monotonic())
            if stop_event.wait(wait_time):
                break

            with segment_lock:
                rows = list(segment_rows)
                segment_rows.clear()

            saved_at = datetime.now()
            timestamp_for_name = saved_at.strftime("%Y%m%d%H%M%S")
            basename = f"{sequence}_{timestamp_for_name}"
            csv_path = OUTPUT_DIR / f"{basename}.csv"
            jpg_path = OUTPUT_DIR / f"{basename}.jpg"

            if rows:
                df = pd.DataFrame(rows, columns=["timestamp", "x", "y", "z"])
                df.to_csv(csv_path, index=False)
            else:
                df = pd.DataFrame(columns=["timestamp", "x", "y", "z"])
                # still save the empty segment so a data-drop window is visible.
                df.to_csv(csv_path, index=False)

            frame = None
            image_src = None
            camera_note = ""

            if camera is not None:
                # discard a couple of frames, then keep the freshest one.
                for _ in range(2):
                    camera.read()
                ok, frame = camera.read()
                if ok and frame is not None:
                    cv2.imwrite(str(jpg_path), frame)
                    image_src = frame_to_data_uri(frame)
                else:
                    camera_note = "Webcam read failed for this segment."
                    jpg_path = None
            else:
                camera_note = "No webcam image: camera unavailable."
                jpg_path = None

            note_parts = []
            if not rows:
                note_parts.append("No complete XYZ rows received in this 10-second window.")
            if camera_note:
                note_parts.append(camera_note)
            note = " ".join(note_parts)

            append_manifest(
                basename,
                csv_path,
                jpg_path,
                len(df),
                note=note,
            )

            with latest_segment_lock:
                latest_segment["version"] += 1
                latest_segment["basename"] = basename
                latest_segment["df"] = df
                latest_segment["image_src"] = image_src
                latest_segment["status"] = (
                    f"Saved {basename}: {len(df)} complete XYZ rows"
                    + (f" | {note}" if note else "")
                )

            print(latest_segment["status"])
            sequence += 1
            next_deadline += CAPTURE_SECONDS

    finally:
        if camera is not None:
            camera.release()


# DASH APP

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H2("SIT225 6D - Accelerometer Activity Capture"),
        html.Div(id="cloud-status", style={"marginBottom": "8px"}),
        dcc.Graph(id="live-graph", figure=empty_live_figure()),
        html.Hr(),
        html.H3("Latest 10-second activity segment"),
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="segment-graph"),
                    style={"width": "62%", "display": "inline-block", "verticalAlign": "top"},
                ),
                html.Div(
                    [
                        html.Img(
                            id="activity-image",
                            style={
                                "maxWidth": "100%",
                                "maxHeight": "430px",
                                "border": "1px solid #aaa",
                            },
                        ),
                        html.P(id="segment-status"),
                    ],
                    style={
                        "width": "36%",
                        "display": "inline-block",
                        "paddingLeft": "2%",
                        "verticalAlign": "top",
                    },
                ),
            ]
        ),
        html.Button("STOP PROGRAM", id="stop-button", n_clicks=0),
        html.Div(id="stop-status", style={"marginTop": "10px"}),
        html.P(
            f"Capture window: {CAPTURE_SECONDS}s | "
            f"Dash check: {DASH_INTERVAL_MS} ms | "
            f"Live graph: ~{int(1 / DISPLAY_INTERVAL_SECONDS)} Hz display, "
            f"{MAX_VISIBLE_POINTS} points (~{MAX_VISIBLE_POINTS * DISPLAY_INTERVAL_SECONDS:.0f}s window)"
        ),
        dcc.Interval(id="ui-tick", interval=DASH_INTERVAL_MS, n_intervals=0),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
)


@app.callback(
    Output("live-graph", "extendData"),
    Input("ui-tick", "n_intervals"),
)
def update_live_graph(_):
    grouped = {a: {"x": [], "y": []} for a in AXES}

    while True:
        try:
            axis, timestamp, value = live_queue.get_nowait()
        except queue.Empty:
            break
        grouped[axis]["x"].append(timestamp)
        grouped[axis]["y"].append(value)

    xs, ys, trace_indices = [], [], []
    for idx, axis in enumerate(AXES):
        if grouped[axis]["x"]:
            xs.append(grouped[axis]["x"])
            ys.append(grouped[axis]["y"])
            trace_indices.append(idx)

    if not trace_indices:
        return no_update

    update = {"x": xs, "y": ys}
    return update, trace_indices, MAX_VISIBLE_POINTS


@app.callback(
    Output("segment-graph", "figure"),
    Output("activity-image", "src"),
    Output("segment-status", "children"),
    Input("ui-tick", "n_intervals"),
)
def update_latest_segment(_):
    with latest_segment_lock:
        basename = latest_segment["basename"]
        df = latest_segment["df"]
        image_src = latest_segment["image_src"]
        status = latest_segment["status"]

    if basename is None or df is None:
        fig = go.Figure()
        fig.update_layout(title="Waiting for first saved segment...")
        return fig, None, status

    fig = make_segment_figure(df, f"{basename}.csv")
    return fig, image_src, status


@app.callback(
    Output("cloud-status", "children"),
    Input("ui-tick", "n_intervals"),
)
def update_cloud_status(_):
    now = datetime.now()
    parts = []
    with callback_lock:
        snapshot = dict(last_callback_time)

    for axis in AXES:
        t = snapshot[axis]
        if t is None:
            parts.append(f"{axis.upper()}: waiting")
        else:
            age = (now - t).total_seconds()
            parts.append(f"{axis.upper()}: {age:.1f}s ago")

    return "Latest direct phone callbacks — " + " | ".join(parts)


@app.callback(
    Output("stop-status", "children"),
    Input("stop-button", "n_clicks"),
    prevent_initial_call=True,
)
def stop_program(n_clicks):
    if not n_clicks:
        return no_update

    stop_event.set()

    def delayed_exit():
        time.sleep(0.8)
        os._exit(0)

    threading.Thread(target=delayed_exit, daemon=True).start()
    return "Stop requested. Finalising capture and closing program..."


# STARTUP

def request_stop(*_):
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    phone_thread = threading.Thread(target=phone_worker, daemon=True)
    phone_thread.start()

    capture_thread = threading.Thread(target=capture_worker, daemon=True)
    capture_thread.start()

    print(f"Direct phone listener started: {PHYPHOX_URL}")
    print(f"Saving 10-second segments to: {OUTPUT_DIR.resolve()}")
    print("Open the Dash URL shown below and collect >30 minutes of balanced activities.")

    # use_reloader=False prevents the program from starting a duplicate capture worker.
    app.run(debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
