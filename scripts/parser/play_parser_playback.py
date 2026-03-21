# scripts/parser/play_parser_playback.py
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import open3d as o3d

from scripts.common.io_paths import resolve_repo_path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay parser playback NPZ built from *_parts_colored.ply and *_part_boxes_lineset.ply.",
    )
    parser.add_argument("--input", type=str, required=True, help="Playback .npz path.")
    parser.add_argument("--fps", type=float, default=10.0, help="Playback FPS.")
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (inclusive). -1 means last frame.")
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--window-name", type=str, default="scan2sim parser playback")
    parser.add_argument("--bg", type=float, nargs=3, default=[0.02, 0.02, 0.02], help="Background RGB [0..1].")
    parser.add_argument("--headless", action="store_true", help="Run without Open3D GUI window.")
    parser.add_argument("--export-dir", type=str, default=None, help="Optional output dir to export playback frames as PLY.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-frame log output.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print per-frame log every N frames (0 disables periodic frame logs).",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=5,
        help="Frame jump size for Up/Down arrow keys in GUI mode.",
    )
    return parser.parse_args(argv)


def _frame_bounds(frame_offsets: np.ndarray, frame_idx: int) -> tuple[int, int]:
    start = int(frame_offsets[frame_idx])
    end = int(frame_offsets[frame_idx + 1])
    return start, end


def _frame_name(frame_names: np.ndarray, frame_idx: int) -> str:
    if frame_idx < int(frame_names.shape[0]):
        return str(frame_names[frame_idx])
    return f"frame_{frame_idx:04d}"


def _export_frames_as_ply(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    frame_offsets: np.ndarray,
    frame_names: np.ndarray,
    frame_indices: Sequence[int],
    export_dir: Path,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in frame_indices:
        lo, hi = _frame_bounds(frame_offsets, frame_idx)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[lo:hi])
        cloud.colors = o3d.utility.Vector3dVector(colors[lo:hi])
        out_path = export_dir / f"{_frame_name(frame_names, frame_idx)}.ply"
        o3d.io.write_point_cloud(str(out_path), cloud, write_ascii=False)
    print(f"[play-parser-playback] exported {len(frame_indices)} frames -> {export_dir}")


def _run_headless_loop(
    *,
    frame_indices: Sequence[int],
    frame_names: np.ndarray,
    total_frames: int,
    loop: bool,
    frame_interval: float,
    quiet: bool,
    log_every: int,
) -> None:
    tick = 0
    interval = max(0, int(log_every))
    while True:
        for frame_idx in frame_indices:
            if (not quiet) and (interval > 0) and (tick % interval == 0):
                frame_name = _frame_name(frame_names, frame_idx)
                print(f"[play-parser-playback][headless] frame={frame_idx}/{total_frames - 1} name={frame_name}")
            time.sleep(frame_interval)
            tick += 1
        if not loop:
            return


def _resolve_component_counts(
    *,
    payload: np.lib.npyio.NpzFile,
    frame_offsets: np.ndarray,
    total_frames: int,
    count_key: str,
    source_key: str,
) -> np.ndarray | None:
    if count_key in payload.files:
        counts = np.asarray(payload[count_key], dtype=np.int64)
        if counts.ndim == 1 and counts.shape[0] == total_frames:
            for frame_idx in range(total_frames):
                frame_total = int(frame_offsets[frame_idx + 1] - frame_offsets[frame_idx])
                value = int(counts[frame_idx])
                if value < 0 or value > frame_total:
                    return None
            return counts
    if source_key not in payload.files:
        return None

    source_paths = np.asarray(payload[source_key])
    if source_paths.ndim != 1 or source_paths.shape[0] != total_frames:
        return None

    counts = np.zeros((total_frames,), dtype=np.int64)
    for frame_idx, raw_path in enumerate(source_paths):
        path = Path(str(raw_path))
        if not path.exists() or (not path.is_file()):
            counts[frame_idx] = 0
            continue
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points)
        count = int(points.shape[0]) if points.ndim == 2 and points.shape[1] == 3 else 0
        frame_total = int(frame_offsets[frame_idx + 1] - frame_offsets[frame_idx])
        if count < 0 or count > frame_total:
            return None
        counts[frame_idx] = count
    print(f"[play-parser-playback] '{count_key}' inferred from '{source_key}'.")
    return counts


@dataclass
class _GuiPlaybackState:
    frame_pos: int
    paused: bool
    loop: bool
    running: bool
    show_part_colors: bool
    show_boxes: bool
    show_markers: bool
    last_tick_ts: float
    tick_accum: float
    tick_counter: int


def _run_gui_loop(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    frame_offsets: np.ndarray,
    frame_part_counts: np.ndarray,
    frame_raw_counts: np.ndarray,
    frame_line_counts: np.ndarray,
    frame_marker_counts: np.ndarray,
    frame_names: np.ndarray,
    frame_indices: Sequence[int],
    total_frames: int,
    frame_interval: float,
    loop: bool,
    window_name: str,
    window_width: int,
    window_height: int,
    bg: Sequence[float],
    quiet: bool,
    log_every: int,
    step_size: int,
) -> bool:
    gui = o3d.visualization.gui
    rendering = o3d.visualization.rendering
    app = gui.Application.instance
    try:
        app.initialize()
    except Exception:
        return False

    window = app.create_window(
        str(window_name),
        max(320, int(window_width)),
        max(240, int(window_height)),
    )
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([float(bg[0]), float(bg[1]), float(bg[2]), 1.0])
    scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)
    window.add_child(scene_widget)
    has_part_points = any(int(frame_part_counts[frame_idx]) > 0 for frame_idx in frame_indices)
    has_raw_points = any(int(frame_raw_counts[frame_idx]) > 0 for frame_idx in frame_indices)
    has_box_points = any(int(frame_line_counts[frame_idx]) > 0 for frame_idx in frame_indices)
    has_marker_points = any(int(frame_marker_counts[frame_idx]) > 0 for frame_idx in frame_indices)

    hud_frame = gui.Label("")
    hud_frame.text_color = gui.Color(1.0, 1.0, 1.0)
    hud_frame.background_color = gui.Color(0.0, 0.0, 0.0, 0.70)
    window.add_child(hud_frame)

    help_text = "Keys: ←/→ step, ↑/↓ jump, Space play/pause, Home/End"
    if has_part_points:
        if has_raw_points:
            help_text += ", C part-color/raw"
        else:
            help_text += ", C part-color on/off"
    if has_box_points:
        help_text += ", B box on/off"
    if has_marker_points:
        help_text += ", M marker on/off"
    hud_help = gui.Label(help_text)
    hud_help.text_color = gui.Color(0.95, 0.95, 0.95)
    hud_help.background_color = gui.Color(0.0, 0.0, 0.0, 0.55)
    window.add_child(hud_help)

    def _on_layout(ctx: gui.LayoutContext) -> None:
        content = window.content_rect
        scene_widget.frame = content
        pad = 10
        pref_frame = hud_frame.calc_preferred_size(ctx, gui.Widget.Constraints())
        pref_help = hud_help.calc_preferred_size(ctx, gui.Widget.Constraints())
        hud_frame.frame = gui.Rect(content.x + pad, content.y + pad, pref_frame.width + 12, pref_frame.height + 8)
        hud_help.frame = gui.Rect(
            content.x + pad,
            content.y + pad + pref_frame.height + 14,
            pref_help.width + 12,
            pref_help.height + 8,
        )

    window.set_on_layout(_on_layout)

    cloud = o3d.geometry.PointCloud()
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    geom_name = "playback_cloud"
    interval = max(0, int(log_every))
    jump = max(1, int(step_size))
    state = _GuiPlaybackState(
        frame_pos=0,
        paused=False,
        loop=bool(loop),
        running=True,
        show_part_colors=True,
        show_boxes=True,
        show_markers=True,
        last_tick_ts=time.monotonic(),
        tick_accum=0.0,
        tick_counter=0,
    )

    def _set_hud() -> None:
        current_frame_idx = int(frame_indices[state.frame_pos])
        current_name = _frame_name(frame_names, current_frame_idx)
        mode = "PAUSE" if state.paused else "PLAY"
        if has_part_points:
            if has_raw_points:
                part_state = "PART:COLOR" if state.show_part_colors else "PART:RAW"
            else:
                part_state = "PART:ON" if state.show_part_colors else "PART:OFF"
        else:
            part_state = "PART:N/A"
        box_state = "BOX:ON" if (has_box_points and state.show_boxes) else ("BOX:OFF" if has_box_points else "BOX:N/A")
        marker_state = (
            "MARKER:ON" if (has_marker_points and state.show_markers) else ("MARKER:OFF" if has_marker_points else "MARKER:N/A")
        )
        hud_frame.text = (
            f"{mode} | {part_state} | {box_state} | {marker_state} | "
            f"frame {current_frame_idx}/{total_frames - 1} | {current_name}"
        )
        window.post_redraw()

    def _render_current_frame(*, log_frame: bool) -> None:
        current_frame_idx = int(frame_indices[state.frame_pos])
        lo_full, hi_full = _frame_bounds(frame_offsets, current_frame_idx)
        frame_total = int(max(0, hi_full - lo_full))
        part_count = int(max(0, min(int(frame_part_counts[current_frame_idx]), frame_total)))
        rem = max(0, frame_total - part_count)
        raw_count = int(max(0, min(int(frame_raw_counts[current_frame_idx]), rem)))
        rem -= raw_count
        line_count = int(max(0, min(int(frame_line_counts[current_frame_idx]), rem)))
        rem -= line_count
        marker_count = int(max(0, min(int(frame_marker_counts[current_frame_idx]), rem)))
        rem -= marker_count
        if rem > 0:
            line_count += rem

        part_lo, part_hi = lo_full, lo_full + part_count
        raw_lo, raw_hi = part_hi, part_hi + raw_count
        line_lo, line_hi = raw_hi, raw_hi + line_count
        marker_lo, marker_hi = line_hi, line_hi + marker_count

        selected_pts = []
        selected_cols = []
        if has_part_points:
            if state.show_part_colors:
                if part_count > 0:
                    selected_pts.append(points[part_lo:part_hi])
                    selected_cols.append(colors[part_lo:part_hi])
            elif has_raw_points and raw_count > 0:
                selected_pts.append(points[raw_lo:raw_hi])
                selected_cols.append(colors[raw_lo:raw_hi])
            elif part_count > 0:
                selected_pts.append(points[part_lo:part_hi])
                selected_cols.append(np.tile(np.array([[0.70, 0.70, 0.70]], dtype=np.float64), (part_count, 1)))
        if has_box_points and state.show_boxes and line_count > 0:
            selected_pts.append(points[line_lo:line_hi])
            selected_cols.append(colors[line_lo:line_hi])
        if has_marker_points and state.show_markers and marker_count > 0:
            selected_pts.append(points[marker_lo:marker_hi])
            selected_cols.append(colors[marker_lo:marker_hi])

        if selected_pts:
            view_pts = np.concatenate(selected_pts, axis=0)
            view_cols = np.concatenate(selected_cols, axis=0)
        else:
            view_pts = np.zeros((0, 3), dtype=np.float64)
            view_cols = np.zeros((0, 3), dtype=np.float64)

        cloud.points = o3d.utility.Vector3dVector(view_pts)
        cloud.colors = o3d.utility.Vector3dVector(view_cols)
        if scene_widget.scene.has_geometry(geom_name):
            scene_widget.scene.remove_geometry(geom_name)
        scene_widget.scene.add_geometry(geom_name, cloud, material)
        if state.tick_counter == 0:
            scene_widget.setup_camera(60.0, scene_widget.scene.bounding_box, scene_widget.scene.bounding_box.get_center())
        if log_frame and (not quiet):
            if interval == 0 or (state.tick_counter % interval == 0):
                current_name = _frame_name(frame_names, current_frame_idx)
                print(f"[play-parser-playback] frame={current_frame_idx}/{total_frames - 1} name={current_name}")
        _set_hud()

    def _step_frame(delta: int, *, from_autoplay: bool) -> bool:
        if delta == 0:
            return True
        next_pos = state.frame_pos + int(delta)
        if 0 <= next_pos < len(frame_indices):
            state.frame_pos = next_pos
            return True
        if state.loop:
            state.frame_pos = next_pos % len(frame_indices)
            return True
        if from_autoplay:
            state.paused = True
        state.frame_pos = max(0, min(next_pos, len(frame_indices) - 1))
        return False

    _render_current_frame(log_frame=True)

    def _on_key(event: gui.KeyEvent) -> bool:
        if event.type != gui.KeyEvent.Type.DOWN:
            return False
        key = event.key
        if key == gui.KeyName.SPACE:
            state.paused = not state.paused
            _set_hud()
            return True
        if key == gui.KeyName.RIGHT:
            state.paused = True
            _step_frame(1, from_autoplay=False)
            _render_current_frame(log_frame=False)
            return True
        if key == gui.KeyName.LEFT:
            state.paused = True
            _step_frame(-1, from_autoplay=False)
            _render_current_frame(log_frame=False)
            return True
        if key == gui.KeyName.UP:
            state.paused = True
            _step_frame(jump, from_autoplay=False)
            _render_current_frame(log_frame=False)
            return True
        if key == gui.KeyName.DOWN:
            state.paused = True
            _step_frame(-jump, from_autoplay=False)
            _render_current_frame(log_frame=False)
            return True
        if key == gui.KeyName.HOME:
            state.paused = True
            state.frame_pos = 0
            _render_current_frame(log_frame=False)
            return True
        if key == gui.KeyName.END:
            state.paused = True
            state.frame_pos = len(frame_indices) - 1
            _render_current_frame(log_frame=False)
            return True
        if key == gui.KeyName.B:
            if has_box_points:
                state.show_boxes = not state.show_boxes
                _render_current_frame(log_frame=False)
            return True
        if key == gui.KeyName.C:
            if has_part_points:
                state.show_part_colors = not state.show_part_colors
                _render_current_frame(log_frame=False)
            return True
        if key == gui.KeyName.M:
            if has_marker_points:
                state.show_markers = not state.show_markers
                _render_current_frame(log_frame=False)
            return True
        return False

    def _on_tick() -> bool:
        if not state.running:
            return False
        now = time.monotonic()
        dt = now - state.last_tick_ts
        state.last_tick_ts = now
        if state.paused:
            return False
        state.tick_accum += dt
        updated = False
        while state.tick_accum >= frame_interval:
            state.tick_accum -= frame_interval
            moved = _step_frame(1, from_autoplay=True)
            if not moved and (not state.loop):
                break
            state.tick_counter += 1
            _render_current_frame(log_frame=True)
            updated = True
        return updated

    def _on_close() -> bool:
        state.running = False
        return True

    window.set_on_key(_on_key)
    window.set_on_tick_event(_on_tick)
    window.set_on_close(_on_close)

    app.run()
    return True


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = repo_root()
    input_path = resolve_repo_path(root, args.input)
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Playback file not found: {input_path}")

    payload = np.load(input_path, allow_pickle=False)
    points = np.asarray(payload["points"], dtype=np.float64)
    colors = np.asarray(payload["colors"], dtype=np.float64)
    frame_offsets = np.asarray(payload["frame_offsets"], dtype=np.int64)
    frame_names = np.asarray(payload["frame_names"]) if "frame_names" in payload.files else np.array([], dtype=np.str_)

    if frame_offsets.ndim != 1 or frame_offsets.shape[0] < 2:
        raise ValueError("Invalid playback file: frame_offsets must have at least 2 elements.")
    total_frames = int(frame_offsets.shape[0] - 1)
    if total_frames <= 0:
        raise ValueError("Playback file contains no frames.")

    start_idx = max(0, int(args.start))
    end_idx = (total_frames - 1) if int(args.end) < 0 else min(int(args.end), total_frames - 1)
    if start_idx > end_idx:
        raise ValueError(f"Invalid frame range: start={start_idx}, end={end_idx}")

    frame_interval = 1.0 / max(float(args.fps), 1e-3)
    frame_indices = list(range(start_idx, end_idx + 1))
    print(
        f"[play-parser-playback] frames={total_frames} range={start_idx}..{end_idx} "
        f"fps={float(args.fps):.3f} loop={bool(args.loop)}"
    )
    if args.export_dir:
        export_dir = resolve_repo_path(root, args.export_dir)
        _export_frames_as_ply(
            points=points,
            colors=colors,
            frame_offsets=frame_offsets,
            frame_names=frame_names,
            frame_indices=frame_indices,
            export_dir=export_dir,
        )
    if args.headless:
        _run_headless_loop(
            frame_indices=frame_indices,
            frame_names=frame_names,
            total_frames=total_frames,
            loop=bool(args.loop),
            frame_interval=frame_interval,
            quiet=bool(args.quiet),
            log_every=int(args.log_every),
        )
        return

    frame_part_counts = _resolve_component_counts(
        payload=payload,
        frame_offsets=frame_offsets,
        total_frames=total_frames,
        count_key="frame_part_counts",
        source_key="source_parts",
    )
    frame_line_counts = _resolve_component_counts(
        payload=payload,
        frame_offsets=frame_offsets,
        total_frames=total_frames,
        count_key="frame_line_counts",
        source_key="source_lines",
    )
    frame_raw_counts = _resolve_component_counts(
        payload=payload,
        frame_offsets=frame_offsets,
        total_frames=total_frames,
        count_key="frame_raw_counts",
        source_key="source_raw",
    )
    frame_marker_counts = _resolve_component_counts(
        payload=payload,
        frame_offsets=frame_offsets,
        total_frames=total_frames,
        count_key="frame_marker_counts",
        source_key="source_markers",
    )

    if frame_part_counts is None:
        frame_part_counts = np.array(
            [int(frame_offsets[i + 1] - frame_offsets[i]) for i in range(total_frames)],
            dtype=np.int64,
        )
    if frame_line_counts is None:
        frame_line_counts = np.zeros((total_frames,), dtype=np.int64)
    if frame_raw_counts is None:
        frame_raw_counts = np.zeros((total_frames,), dtype=np.int64)
    if frame_marker_counts is None:
        frame_marker_counts = np.zeros((total_frames,), dtype=np.int64)

    for frame_idx in range(total_frames):
        frame_total = int(max(0, frame_offsets[frame_idx + 1] - frame_offsets[frame_idx]))
        part_count = int(max(0, min(int(frame_part_counts[frame_idx]), frame_total)))
        rem = frame_total - part_count
        raw_count = int(max(0, min(int(frame_raw_counts[frame_idx]), rem)))
        rem -= raw_count
        line_count = int(max(0, min(int(frame_line_counts[frame_idx]), rem)))
        rem -= line_count
        marker_count = int(max(0, min(int(frame_marker_counts[frame_idx]), rem)))
        rem -= marker_count
        if rem > 0:
            line_count += rem
        frame_part_counts[frame_idx] = part_count
        frame_raw_counts[frame_idx] = raw_count
        frame_line_counts[frame_idx] = line_count
        frame_marker_counts[frame_idx] = marker_count

    gui_ok = _run_gui_loop(
        points=points,
        colors=colors,
        frame_offsets=frame_offsets,
        frame_part_counts=frame_part_counts,
        frame_raw_counts=frame_raw_counts,
        frame_line_counts=frame_line_counts,
        frame_marker_counts=frame_marker_counts,
        frame_names=frame_names,
        frame_indices=frame_indices,
        total_frames=total_frames,
        frame_interval=frame_interval,
        loop=bool(args.loop),
        window_name=str(args.window_name),
        window_width=int(args.window_width),
        window_height=int(args.window_height),
        bg=args.bg,
        quiet=bool(args.quiet),
        log_every=int(args.log_every),
        step_size=int(args.step_size),
    )
    if not gui_ok:
        print("[play-parser-playback] Open3D GUI initialization failed. Falling back to headless mode.")
        _run_headless_loop(
            frame_indices=frame_indices,
            frame_names=frame_names,
            total_frames=total_frames,
            loop=bool(args.loop),
            frame_interval=frame_interval,
            quiet=bool(args.quiet),
            log_every=int(args.log_every),
        )


if __name__ == "__main__":
    main()
