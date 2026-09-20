from pathlib import Path
import csv
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CAPTURE_DIR = Path("captures_6D")
OUTPUT_DIR = Path("6d_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATION_FILE = OUTPUT_DIR / "annotation.csv"
DECISION_LOG = OUTPUT_DIR / "annotation_decision_log.csv"
ANALYSIS_DIR = OUTPUT_DIR / "analysis_outputs"

ACTIVITY_NAMES = {
    0: "no_activity",
    1: "tilt",
    2: "side_to_side_wave",
}


def sort_key(path: Path):
    try:
        return int(path.stem.split("_", 1)[0])
    except ValueError:
        return 10**9


def load_existing_annotations():
    labels = {}
    if ANNOTATION_FILE.exists():
        df = pd.read_csv(ANNOTATION_FILE)
        for _, row in df.iterrows():
            labels[str(row["filename"])] = int(row["activity_label"])
    return labels


def save_annotation(labels):
    rows = sorted(labels.items(), key=lambda kv: sort_key(Path(kv[0])))
    pd.DataFrame(rows, columns=["filename", "activity_label"]).to_csv(
        ANNOTATION_FILE, index=False
    )


def append_decision(filename, label, basis, notes=""):
    new_file = not DECISION_LOG.exists()
    with DECISION_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["filename", "activity_label", "decision_basis", "notes"])
        w.writerow([filename, label, basis, notes])


def show_image_and_graph(image_path: Path, csv_path: Path):
    frame = cv2.imread(str(image_path))
    if frame is None:
        rgb = None
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        df = pd.DataFrame()
        print(f"Could not read {csv_path.name}: {exc}")

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [1, 1.7]}
    )

    # left: activity image
    if rgb is not None:
        axes[0].imshow(rgb)
    else:
        axes[0].text(
            0.5, 0.5, "Image unavailable",
            ha="center", va="center", transform=axes[0].transAxes
        )
    axes[0].set_title(f"Webcam image\n{image_path.name}")
    axes[0].axis("off")

    # right: matching acceleration segment
    if not df.empty and all(col in df.columns for col in ("x", "y", "z")):
        for axis in ("x", "y", "z"):
            axes[1].plot(df.index, df[axis], label=axis.upper(), linewidth=1.2)
        axes[1].legend(loc="upper right")
    else:
        axes[1].text(
            0.5, 0.5, "CSV unavailable or invalid",
            ha="center", va="center", transform=axes[1].transAxes
        )

    axes[1].set_title(f"Matching accelerometer segment\n{csv_path.name}")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Acceleration")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        "0 = no activity   |   1 = tilt   |   2 = side-to-side   |   u = uncertain   |   q = quit",
        fontsize=13
    )
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.2)
    return fig


def wait_for_key(fig, valid):
    pressed = {"key": None}

    def on_key(event):
        key = (event.key or "").lower()
        if key in valid:
            pressed["key"] = key

    connection_id = fig.canvas.mpl_connect("key_press_event", on_key)

    while pressed["key"] is None and plt.fignum_exists(fig.number):
        plt.pause(0.05)

    fig.canvas.mpl_disconnect(connection_id)
    return pressed["key"] or "q"



def annotate():
    labels = load_existing_annotations()

    images = sorted(CAPTURE_DIR.glob("*.jpg"), key=sort_key)
    if not images:
        raise FileNotFoundError(
            f"No JPG captures found in {CAPTURE_DIR.resolve()}. "
            "Make sure your valid captures are in that folder."
        )

    print("\nANNOTATION MODE: image + matching graph are shown for every segment.")
    print("Labels: 0=no activity, 1=tilt, 2=side-to-side wave, u=uncertain, q=quit/save")
    print("Decision basis: i=image, g=graph, b=both")
    print(f"Outputs: {OUTPUT_DIR.resolve()}\n")

    total = len(images)

    for index, image_path in enumerate(images, start=1):
        csv_path = image_path.with_suffix(".csv")
        csv_name = csv_path.name

        if csv_name in labels:
            continue

        if not csv_path.exists():
            print(f"Skipping {image_path.name}: matching CSV is missing.")
            continue

        fig = show_image_and_graph(image_path, csv_path)

        try:
            print(
                f"[{index}/{total}] {csv_name} "
                "press 0, 1, 2, u, or q"
            )
            answer = wait_for_key(fig, {"0", "1", "2", "u", "q"})

            if answer == "q":
                save_annotation(labels)
                plt.close(fig)
                print("Progress saved.")
                return labels, []

            if answer == "u":
                append_decision(
                    csv_name,
                    -1,
                    "uncertain",
                    "Marked uncertain for later review"
                )
                print("Marked uncertain")
                continue

            label = int(answer)

            fig.suptitle(
                "i = image   |   g = graph   |   b = both",
                fontsize=13
            )
            fig.canvas.draw_idle()
            print("Main basis [i=image, g=graph, b=both]")
            basis_answer = wait_for_key(fig, {"i", "g", "b"})

            basis_map = {
                "i": "image",
                "g": "accelerometer_data",
                "b": "image_and_accelerometer_data",
            }
            basis = basis_map[basis_answer]

            labels[csv_name] = label
            append_decision(
                csv_name,
                label,
                basis,
                f"Assigned as {ACTIVITY_NAMES[label]}"
            )
            save_annotation(labels)
        finally:
            plt.close(fig)

    save_annotation(labels)
    return labels, []


def compute_features_for_segment(csv_path: Path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    mag = np.sqrt(x*x + y*y + z*z)

    out = {"filename": csv_path.name, "samples": len(df)}
    for name, arr in [("x", x), ("y", y), ("z", z), ("mag", mag)]:
        out[f"{name}_mean"] = float(np.mean(arr))
        out[f"{name}_std"] = float(np.std(arr, ddof=0))
        out[f"{name}_range"] = float(np.max(arr) - np.min(arr))
        out[f"{name}_rms"] = float(np.sqrt(np.mean(arr * arr)))

    # total variation is a simple movement-intensity measure
    out["xyz_total_variation"] = float(
        np.abs(np.diff(x)).sum()
        + np.abs(np.diff(y)).sum()
        + np.abs(np.diff(z)).sum()
    )
    return out


def build_feature_tables(labels):
    rows = []
    for csv_name, label in labels.items():
        csv_path = CAPTURE_DIR / csv_name
        if not csv_path.exists():
            continue
        features = compute_features_for_segment(csv_path)
        if features is None:
            continue
        features["activity_label"] = label
        features["activity_name"] = ACTIVITY_NAMES[label]
        rows.append(features)

    if not rows:
        print("No usable labelled CSV segments found.")
        return None

    features_df = pd.DataFrame(rows)
    features_df.to_csv(CAPTURE_DIR / "feature_summary.csv", index=False)

    numeric_cols = [
        c for c in features_df.columns
        if c not in ("filename", "activity_name")
        and pd.api.types.is_numeric_dtype(features_df[c])
    ]
    # exclude activity_label from aggregated measurements
    measure_cols = [c for c in numeric_cols if c != "activity_label"]

    summary = (
        features_df
        .groupby(["activity_label", "activity_name"])[measure_cols]
        .agg(["mean", "std", "median"])
        .round(4)
    )
    summary.to_csv(CAPTURE_DIR / "class_feature_summary.csv")
    return features_df


def make_multi_instance_plots(features_df, examples_per_class=3):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    for label, activity_name in ACTIVITY_NAMES.items():
        class_rows = features_df[features_df["activity_label"] == label]
        if class_rows.empty:
            continue
        if len(class_rows) <= examples_per_class:
            selected = class_rows
        else:
            positions = np.linspace(0, len(class_rows) - 1, examples_per_class, dtype=int)
            selected = class_rows.iloc[positions]

        fig, axes = plt.subplots(
            len(selected), 1,
            figsize=(11, 3.2 * len(selected)),
            squeeze=False
        )

        for row_idx, (_, row) in enumerate(selected.iterrows()):
            csv_name = row["filename"]
            csv_path = CAPTURE_DIR / csv_name
            df = pd.read_csv(csv_path)
            ax = axes[row_idx, 0]
            for axis in ("x", "y", "z"):
                ax.plot(df.index, df[axis], label=axis.upper())
            ax.set_title(csv_name)
            ax.set_ylabel("Acceleration")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.25)

        axes[-1, 0].set_xlabel("Sample index")
        fig.suptitle(f"Multiple instances: {activity_name}", y=1.01)
        fig.tight_layout()
        fig.savefig(
            ANALYSIS_DIR / f"label_{label}_{activity_name}_multi_instance_graphs.png",
            dpi=160,
            bbox_inches="tight"
        )
        plt.close(fig)


def make_contact_sheets(features_df, examples_per_class=3):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    for label, activity_name in ACTIVITY_NAMES.items():
        class_rows = features_df[features_df["activity_label"] == label]
        if class_rows.empty:
            continue
        if len(class_rows) <= examples_per_class:
            selected = class_rows
        else:
            positions = np.linspace(0, len(class_rows) - 1, examples_per_class, dtype=int)
            selected = class_rows.iloc[positions]

        fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4.5), squeeze=False)

        for idx, (_, row) in enumerate(selected.iterrows()):
            csv_name = row["filename"]
            image_path = CAPTURE_DIR / Path(csv_name).with_suffix(".jpg").name
            bgr = cv2.imread(str(image_path))
            ax = axes[0, idx]
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(rgb)
            ax.set_title(image_path.name)
            ax.axis("off")

        fig.suptitle(f"Activity images: {activity_name}", y=1.02)
        fig.tight_layout()
        fig.savefig(
            ANALYSIS_DIR / f"label_{label}_{activity_name}_activity_images.png",
            dpi=160,
            bbox_inches="tight"
        )
        plt.close(fig)


def main():
    labels, _ = annotate()
    cv2.destroyAllWindows()

    features_df = build_feature_tables(labels)
    if features_df is not None:
        make_multi_instance_plots(features_df)
        make_contact_sheets(features_df)
        print("\nCreated:")
        print(f"- {ANNOTATION_FILE}")
        print(f"- {DECISION_LOG}")
        print(f"- {CAPTURE_DIR / 'feature_summary.csv'}")
        print(f"- {CAPTURE_DIR / 'class_feature_summary.csv'}")
        print(f"- {ANALYSIS_DIR}/*.png")


if __name__ == "__main__":
    main()
