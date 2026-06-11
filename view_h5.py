"""Interactive H5 dataset viewer with segment/view folder selector.

Usage:
  python3 scripts/view_h5.py hand_crops/data.h5
  python3 scripts/view_h5.py hand_crops/01_0519_03_all/bleft/data.h5

Left panel  — click a segment/view folder to filter samples; scroll to browse the list
Navigate    — ← → arrow keys  |  Prev / Next buttons
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.widgets import Button

# ── skeleton drawing ─────────────────────────────────────────────────────────

SKELETON_CONNECTIONS = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

_C = dict(wrist="#ffffff", thumb="#ff5050", index="#50ff50",
          middle="#5050ff", ring="#ffff50", pinky="#ff50ff")
KP_COLORS = [_C["wrist"],
              *[_C["thumb"]] * 4, *[_C["index"]] * 4,
              *[_C["middle"]] * 4, *[_C["ring"]] * 4, *[_C["pinky"]] * 4]
CONN_COLORS = [_C["thumb"], _C["index"], _C["middle"], _C["ring"], _C["pinky"],
               *[_C["thumb"]] * 3, *[_C["index"]] * 3,
               *[_C["middle"]] * 3, *[_C["ring"]] * 3, *[_C["pinky"]] * 3]


def draw_skeleton(ax, kp2d):
    for idx, (i, j) in enumerate(SKELETON_CONNECTIONS):
        ax.plot([kp2d[i, 0], kp2d[j, 0]], [kp2d[i, 1], kp2d[j, 1]],
                color=CONN_COLORS[idx], linewidth=1.5, solid_capstyle="round")
    ax.scatter(kp2d[:, 0], kp2d[:, 1], c=KP_COLORS, s=30, zorder=5, edgecolors="none")


# ── scrollable folder list ───────────────────────────────────────────────────

class FolderList:
    """Clickable + scrollable list of segment/view folders in a matplotlib axes."""

    _BG       = "#0d0d1a"
    _FG       = "#7777aa"
    _SEL_FG   = "#ffff88"
    _HOV_FG   = "#aaaadd"
    _VISIBLE  = 22          # max rows shown at once

    def __init__(self, ax, folders: list[str], on_select):
        """
        folders: list of folder labels, first entry should be "All".
        on_select: callable(label: str)
        """
        self.ax        = ax
        self.folders   = folders
        self.on_select = on_select
        self.offset    = 0
        self.selected  = 0   # index into self.folders

        ax.set_facecolor(self._BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, self._VISIBLE)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")

        ax.set_title("Folder", color="#888899", fontsize=8, pad=3)

        self._texts = []
        for row in range(self._VISIBLE):
            t = ax.text(0.06, self._VISIBLE - row - 0.5, "",
                        va="center", fontsize=7.5, fontfamily="monospace",
                        color=self._FG, clip_on=True)
            self._texts.append(t)

        # scroll indicator bar
        self._scroll_bg  = ax.axvspan(0.93, 1.0, ymin=0, ymax=1,
                                       color="#1a1a33", zorder=0)
        self._scroll_bar = ax.axvspan(0.93, 1.0, ymin=0.9, ymax=1.0,
                                       color="#4444aa", zorder=1)

        fig = ax.figure
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("scroll_event",       self._on_scroll)

        self._render()

    # ── internal ────────────────────────────────────────────────────────────

    def _render(self):
        n = len(self.folders)
        visible_slice = self.folders[self.offset: self.offset + self._VISIBLE]
        for row, t in enumerate(self._texts):
            gi = self.offset + row   # global index
            if row < len(visible_slice):
                label = visible_slice[row]
                if gi == self.selected:
                    color, weight = self._SEL_FG, "bold"
                    prefix = "▶ "
                else:
                    color, weight = self._FG, "normal"
                    prefix = "  "
                # Truncate long names
                display = (prefix + label)[:34]
                t.set_text(display); t.set_color(color)
                t.set_fontweight(weight); t.set_visible(True)
            else:
                t.set_visible(False)

        # Update scroll indicator
        if n > self._VISIBLE:
            bar_h  = self._VISIBLE / n
            bar_y0 = 1.0 - (self.offset + self._VISIBLE) / n
            bar_y1 = bar_y0 + bar_h
            self._scroll_bar.set_xy([[0.93, bar_y0 * self._VISIBLE],
                                      [0.93, bar_y1 * self._VISIBLE],
                                      [1.00, bar_y1 * self._VISIBLE],
                                      [1.00, bar_y0 * self._VISIBLE]])
        self.ax.figure.canvas.draw_idle()

    def _axes_row(self, event):
        """Return which list row (0-based) the mouse event is on, or None."""
        ax = self.ax
        if event.inaxes != ax:
            return None
        # event.ydata is in axes data coords (0..VISIBLE)
        row = int(self._VISIBLE - event.ydata - 0.001)
        if 0 <= row < self._VISIBLE:
            return row
        return None

    def _on_click(self, event):
        if event.button != 1:
            return
        row = self._axes_row(event)
        if row is None:
            return
        gi = self.offset + row
        if 0 <= gi < len(self.folders):
            self.selected = gi
            self._render()
            self.on_select(self.folders[gi])

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        n = len(self.folders)
        max_off = max(0, n - self._VISIBLE)
        if event.button == "up":
            self.offset = max(0, self.offset - 1)
        elif event.button == "down":
            self.offset = min(max_off, self.offset + 1)
        self._render()

    def jump_to(self, idx: int):
        """Programmatically select an index."""
        self.selected = max(0, min(idx, len(self.folders) - 1))
        # Scroll so selected is visible
        if self.selected < self.offset:
            self.offset = self.selected
        elif self.selected >= self.offset + self._VISIBLE:
            self.offset = self.selected - self._VISIBLE + 1
        self._render()


# ── main viewer ──────────────────────────────────────────────────────────────

class H5Viewer:

    _BG  = "#10101e"
    _AXB = "#0d0d1a"

    def __init__(self, h5_path: Path, start: int = 0):
        self.f = h5py.File(h5_path, "r")
        self.N = int(self.f.attrs.get("n_samples", len(self.f["frame_keys"])))

        kn = self.f.attrs.get("keypoint_names", "[]")
        self.kp_names = json.loads(kn) if isinstance(kn, str) else list(kn)

        # ── build folder list ─────────────────────────────────────────────
        # Unique "segment/view" combinations (or just "view" if single-segment H5)
        has_segment = "segment" in self.f and "view" in self.f
        if has_segment:
            segs  = [self._dec(s) for s in self.f["segment"]]
            views = [self._dec(v) for v in self.f["view"]]
            combos = list(dict.fromkeys(f"{s}/{v}" for s, v in zip(segs, views)))
        else:
            views = [self._dec(v) for v in self.f.get("view",
                     ["unknown"] * self.N)]
            combos = list(dict.fromkeys(views))

        self._folders   = ["All"] + combos
        self._all_idx   = list(range(self.N))   # current filtered indices
        self._filter    = "All"

        # ── layout ───────────────────────────────────────────────────────
        # cols: [folder list] [crop] [crop+skel] [mask] [skeleton] [uv]
        self.fig = plt.figure(figsize=(20, 8.5), facecolor=self._BG)
        self.fig.canvas.manager.set_window_title(f"H5 Viewer — {h5_path.name}")

        gs = gridspec.GridSpec(
            2, 6,
            figure=self.fig,
            width_ratios=[1.6, 1, 1, 1, 1, 1],
            height_ratios=[7, 1],
            hspace=0.06, wspace=0.03,
            left=0.01, right=0.995, top=0.94, bottom=0.01,
        )

        # Folder list spans both rows on col 0
        ax_list = self.fig.add_subplot(gs[:, 0])
        self.folder_list = FolderList(ax_list, self._folders,
                                      on_select=self._on_folder_select)

        # Image panels
        titles = ["Crop", "Crop + Skeleton", "Mask", "Skeleton", "UV"]
        self._img_axes = []
        for col, title in enumerate(titles, start=1):
            ax = self.fig.add_subplot(gs[0, col])
            ax.set_facecolor(self._AXB)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor("#333355")
            ax.set_title(title, color="#9999bb", fontsize=9, pad=3)
            self._img_axes.append(ax)

        # Info text (cols 1-3)
        ax_info = self.fig.add_subplot(gs[1, 1:4])
        ax_info.set_axis_off()
        self._txt = ax_info.text(0.01, 0.5, "", transform=ax_info.transAxes,
                                  color="#aaaacc", fontsize=8.5, va="center",
                                  fontfamily="monospace")

        # Prev / Next
        ax_prev = self.fig.add_subplot(gs[1, 4])
        ax_next = self.fig.add_subplot(gs[1, 5])
        for ax, label, delta in [(ax_prev, "◀  Prev", -1), (ax_next, "Next  ▶", +1)]:
            btn = Button(ax, label, color="#1e1e3a", hovercolor="#3a3a6a")
            btn.label.set_color("#ccccee"); btn.label.set_fontsize(9)
            btn.on_clicked(lambda _, d=delta: self._step(d))

        self._counter = self.fig.suptitle("", color="#7777aa", fontsize=10, y=0.975)
        self._im_handles = [None] * 5

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Set initial local index
        self._loc_idx = max(0, min(start, self.N - 1))
        self._draw()
        plt.show()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _dec(b) -> str:
        return b.decode() if isinstance(b, bytes) else str(b)

    def _filtered(self) -> list[int]:
        if self._filter == "All":
            return self._all_idx
        f = self.f
        has_seg = "segment" in f and "view" in f
        parts = self._filter.split("/")
        result = []
        for gi in self._all_idx:
            if has_seg:
                seg  = self._dec(f["segment"][gi])
                view = self._dec(f["view"][gi])
                label = f"{seg}/{view}"
            else:
                label = self._dec(f["view"][gi])
            if label == self._filter:
                result.append(gi)
        return result

    def _global_idx(self) -> int:
        filt = self._filtered()
        if not filt:
            return 0
        return filt[max(0, min(self._loc_idx, len(filt) - 1))]

    # ── event handlers ───────────────────────────────────────────────────────

    def _on_folder_select(self, label: str):
        self._filter  = label
        self._loc_idx = 0
        self._draw()

    def _step(self, delta: int):
        filt = self._filtered()
        if not filt:
            return
        self._loc_idx = (self._loc_idx + delta) % len(filt)
        self._draw()

    def _on_key(self, event):
        if event.key in ("right", "n", "d"):   self._step(+1)
        elif event.key in ("left", "p", "a"):  self._step(-1)
        elif event.key == "q":                  plt.close(self.fig)

    # ── draw ─────────────────────────────────────────────────────────────────

    def _draw(self):
        filt = self._filtered()
        if not filt:
            self._counter.set_text("No samples for this selection")
            self.fig.canvas.draw_idle()
            return

        gi   = self._global_idx()
        f    = self.f

        crop     = f["crops"][gi]
        mask     = f["masks"][gi]
        skeleton = f["skeletons"][gi]
        uv       = f["uvs"][gi]
        kp2d     = f["keypoints_2d_output"][gi]

        key       = self._dec(f["frame_keys"][gi])
        side      = self._dec(f["side"][gi])
        frame_idx = int(f["frame_idx"][gi])
        is_right  = int(f["is_right"][gi])
        bbox      = f["bbox"][gi].tolist()
        seg_str   = self._dec(f["segment"][gi]) if "segment" in f else ""
        view_str  = self._dec(f["view"][gi])    if "view"    in f else ""

        images = [crop, crop, mask, skeleton, uv]
        modes  = ["RGB", "RGB", "L", "RGB", "RGB"]

        for col, (arr, mode) in enumerate(zip(images, modes)):
            ax = self._img_axes[col]
            if self._im_handles[col] is None:
                kwargs = dict(interpolation="nearest")
                if mode == "L":
                    kwargs.update(cmap="gray", vmin=0, vmax=255)
                self._im_handles[col] = ax.imshow(arr, **kwargs)
            else:
                self._im_handles[col].set_data(arr)

        # Overlay skeleton on crop+skel panel (col index 1)
        ax_skel = self._img_axes[1]
        while ax_skel.lines:       ax_skel.lines[0].remove()
        while ax_skel.collections: ax_skel.collections[0].remove()
        draw_skeleton(ax_skel, kp2d)

        # Info line
        loc = f"{self._loc_idx + 1}/{len(filt)}"
        folder_label = f"{seg_str}/{view_str}" if seg_str else view_str
        self._txt.set_text(
            f"{key}  |  folder: {folder_label}  |  side: {side} "
            f"({'R' if is_right else 'L'})  |  frame: {frame_idx}  |  bbox: {bbox}"
        )
        self._counter.set_text(
            f"{loc}  (global {gi + 1}/{self.N})  —  {self._filter}"
        )

        self.fig.canvas.draw_idle()

    def __del__(self):
        try: self.f.close()
        except Exception: pass


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5", type=Path, help="Path to data.h5")
    parser.add_argument("--start", type=int, default=0,
                        help="Local sample index within current folder to start at")
    args = parser.parse_args()

    if not args.h5.exists():
        print(f"File not found: {args.h5}", file=sys.stderr)
        sys.exit(1)

    H5Viewer(args.h5, start=args.start)


if __name__ == "__main__":
    main()
