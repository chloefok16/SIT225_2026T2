import csv
import getpass
import logging
import os
import queue
import signal
import threading
import time
from datetime import datetime

from arduino_iot_cloud import ArduinoCloudClient
from dash import Dash, Input, Output, dcc, html, no_update
import plotly.graph_objects as go


# settings
CSV_FILE = "live_accelerometer_xyz.csv"
DASH_REFRESH_MS = 250      
MAX_GRAPH_POINTS = 300
PORT = 8051


# shared state
graph_queue = queue.Queue(maxsize=5000)

latest = {"x": None, "y": None, "z": None}
fresh_axes = set()
state_lock = threading.Lock()
csv_lock = threading.Lock()

last_received = {"x": None, "y": None, "z": None}
last_received_lock = threading.Lock()

cloud_status = "Waiting to connect to Arduino Cloud..."
status_lock = threading.Lock()


# logging cleanup
class ArduinoNoiseFilter(logging.Filter):
    """Hide a few non-fatal library messages that make the terminal unreadable."""

    def filter(self, record):
        msg = record.getMessage()
        noisy_fragments = (
            "connection_task raised exception",
            "discovery raised exception",
        )
        return not any(fragment in msg for fragment in noisy_fragments)


def configure_logging():
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    logging.getLogger("dash").setLevel(logging.ERROR)
    logging.getLogger("flask").setLevel(logging.ERROR)

    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(ArduinoNoiseFilter())


# utilities
def set_cloud_status(message):
    global cloud_status
    with status_lock:
        cloud_status = message
    print(message)


def initialise_csv():
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "x", "y", "z"])


def write_csv_row(timestamp, x, y, z):
    with csv_lock:
        with open(CSV_FILE, "a", newline="") as file:
            csv.writer(file).writerow([timestamp, x, y, z])


def put_graph_event(axis, value, timestamp):
    """
    Queue one axis update immediately.

    If the queue fills, remove the oldest event so the display stays close
    to 'now' instead of building a large backlog.
    """
    event = {
        "axis": axis,
        "value": value,
        "time": timestamp,
    }

    try:
        graph_queue.put_nowait(event)
    except queue.Full:
        try:
            graph_queue.get_nowait()
        except queue.Empty:
            pass
        graph_queue.put_nowait(event)


def handle_axis_update(axis, value):
    """
    Called whenever ONE cloud accelerometer variable changes.

    Important:
    - Graph event is queued immediately for low visual latency.
    - CSV row still waits for a fresh X + Y + Z set, preserving the
      combined Week 8 data format.
    """
    now = datetime.now()
    timestamp = now.isoformat(timespec="milliseconds")

    # 1) send this axis to the graph immediately.
    put_graph_event(axis, value, timestamp)

    with last_received_lock:
        last_received[axis] = now

    # 2) maintain the combined X/Y/Z CSV separately.
    complete_sample = None

    with state_lock:
        latest[axis] = value
        fresh_axes.add(axis)

        if fresh_axes == {"x", "y", "z"}:
            complete_sample = (
                timestamp,
                latest["x"],
                latest["y"],
                latest["z"],
            )
            fresh_axes.clear()

    if complete_sample:
        write_csv_row(*complete_sample)


# arduino callbacks
def on_x_changed(client, value):
    handle_axis_update("x", value)


def on_y_changed(client, value):
    handle_axis_update("y", value)


def on_z_changed(client, value):
    handle_axis_update("z", value)


def run_arduino_cloud(device_id, secret_key):
    set_cloud_status("Connecting to Arduino IoT Cloud...")

    client = ArduinoCloudClient(
        device_id=device_id,
        username=device_id,
        password=secret_key,
    )

    client.register("accel_x", value=None, on_write=on_x_changed)
    client.register("accel_y", value=None, on_write=on_y_changed)
    client.register("accel_z", value=None, on_write=on_z_changed)

    try:
        set_cloud_status("Arduino Cloud listener running.")
        client.start()
    except Exception as exc:
        set_cloud_status(
            f"Arduino Cloud listener stopped: {type(exc).__name__}: {exc}"
        )


# wrapper function
def attach_smooth_stream(
    app,
    graph_id,
    interval_id,
    data_queue,
    series_names=("x", "y", "z"),
    max_points=300,
):

    axis_to_trace = {name: i for i, name in enumerate(series_names)}

    @app.callback(
        Output(graph_id, "extendData"),
        Input(interval_id, "n_intervals"),
    )
    def push_new_points(_):
        events = []

        while True:
            try:
                events.append(data_queue.get_nowait())
            except queue.Empty:
                break

        if not events:
            return no_update

        # group only axes that actually received new values.
        grouped = {axis: {"x": [], "y": []} for axis in series_names}

        for event in events:
            axis = event["axis"]
            if axis in grouped:
                grouped[axis]["x"].append(event["time"])
                grouped[axis]["y"].append(event["value"])

        active_axes = [
            axis for axis in series_names
            if grouped[axis]["x"]
        ]

        if not active_axes:
            return no_update

        update = {
            "x": [grouped[axis]["x"] for axis in active_axes],
            "y": [grouped[axis]["y"] for axis in active_axes],
        }

        trace_indexes = [axis_to_trace[axis] for axis in active_axes]

        return [update, trace_indexes, max_points]


# dash app
def build_dash_app():
    app = Dash(__name__)

    figure = go.Figure()

    for axis_name in ("X", "Y", "Z"):
        figure.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name=axis_name,
            )
        )

    figure.update_layout(
        title="Live Smartphone Accelerometer",
        xaxis_title="Time received by Python",
        yaxis_title="Acceleration",
        hovermode="x unified",
        uirevision="keep-view",
        margin=dict(l=60, r=30, t=70, b=60),
    )

    app.layout = html.Div(
        [
            html.H2("SIT225 Live Accelerometer Monitor"),

            html.Div(
                id="cloud-status",
                children="Starting...",
                style={"marginBottom": "6px"},
            ),

            html.Div(
                id="live-age",
                children="Waiting for sensor data...",
                style={"marginBottom": "12px"},
            ),

            dcc.Graph(
                id="live-graph",
                figure=figure,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                },
            ),

            dcc.Interval(
                id="graph-timer",
                interval=DASH_REFRESH_MS,
                n_intervals=0,
            ),

            dcc.Interval(
                id="status-timer",
                interval=1000,
                n_intervals=0,
            ),

            html.Button(
                "STOP PROGRAM",
                id="stop-button",
                n_clicks=0,
                style={
                    "fontSize": "16px",
                    "padding": "10px 18px",
                    "cursor": "pointer",
                    "marginTop": "10px",
                },
            ),

            html.Div(
                id="stop-message",
                style={"marginTop": "8px"},
            ),

            html.P(
                f"CSV logging: {CSV_FILE} | "
                f"Dashboard check: {DASH_REFRESH_MS} ms | "
                f"Rolling window: {MAX_GRAPH_POINTS} points"
            ),
        ],
        style={
            "maxWidth": "1100px",
            "margin": "30px auto",
            "fontFamily": "Arial, sans-serif",
        },
    )

    attach_smooth_stream(
        app=app,
        graph_id="live-graph",
        interval_id="graph-timer",
        data_queue=graph_queue,
        series_names=("x", "y", "z"),
        max_points=MAX_GRAPH_POINTS,
    )

    @app.callback(
        Output("cloud-status", "children"),
        Input("status-timer", "n_intervals"),
    )
    def refresh_cloud_status(_):
        with status_lock:
            return cloud_status

    @app.callback(
        Output("live-age", "children"),
        Input("status-timer", "n_intervals"),
    )
    def refresh_live_age(_):
        now = datetime.now()

        with last_received_lock:
            ages = {}
            for axis in ("x", "y", "z"):
                received = last_received[axis]
                ages[axis] = (
                    None if received is None
                    else (now - received).total_seconds()
                )

        if all(age is None for age in ages.values()):
            return "Waiting for first accelerometer values..."

        parts = []
        for axis in ("x", "y", "z"):
            age = ages[axis]
            if age is None:
                parts.append(f"{axis.upper()}: waiting")
            else:
                parts.append(f"{axis.upper()}: {age:.1f}s ago")

        return "Latest cloud callbacks — " + " | ".join(parts)

    @app.callback(
        Output("stop-message", "children"),
        Input("stop-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def stop_from_browser(_):
        def exit_shortly():
            time.sleep(0.5)
            os._exit(0)

        threading.Thread(target=exit_shortly, daemon=True).start()
        return "Stopping Python process..."

    return app


# reliable stopping
def install_force_exit_handlers():
    def force_exit(signum, frame):
        print("\nStopping program...")
        os._exit(0)

    signal.signal(signal.SIGINT, force_exit)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, force_exit)


# main
def main():
    configure_logging()

    print("SIT225 live accelerometer dashboard")
    print("-----------------------------------")
    print("Credentials are entered at runtime and are not stored in this file.")
    print()

    device_id = (
        os.getenv("ARDUINO_DEVICE_ID")
        or input("Arduino Device ID: ").strip()
    )

    secret_key = (
        os.getenv("ARDUINO_SECRET_KEY")
        or getpass.getpass("Arduino Secret Key (input hidden): ").strip()
    )

    if not device_id or not secret_key:
        raise RuntimeError("Device ID and secret key are required.")

    initialise_csv()
    install_force_exit_handlers()

    cloud_thread = threading.Thread(
        target=run_arduino_cloud,
        args=(device_id, secret_key),
        daemon=True,
        name="ArduinoCloudThread",
    )
    cloud_thread.start()

    app = build_dash_app()

    print()
    print(f"Dashboard: http://127.0.0.1:{PORT}")
    print("Use STOP PROGRAM in the dashboard to exit.")
    print("Ctrl+C is also configured as a forced-exit fallback.")
    print()

    app.run(
        host="127.0.0.1",
        port=PORT,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
