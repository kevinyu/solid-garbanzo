from collections import defaultdict

import numpy as np
from PyQt5 import QtWidgets as widgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.decomposition import PCA


class TimeseriesPlot(widgets.QFrame):

    ndim = 1
    max_points = 10000
    detail_level_scatter_size = {
        2: 4,
        1: 3,
        None: 2
    }

    def __init__(self, size=(700, 100), parent=None):
        super().__init__(parent)
        self.size = size

        self._rotation_period = 100

        self.setup_detail_level_selector()
        self.setup_plots()
        self.setup_data()
        self.init_ui()
        self.parent().UPDATED_CLUSTERS.connect(self.reset)
        self.parent().CLUSTER_HIGHLIGHT.connect(self.on_cluster_highlight)
        self.parent().CLUSTER_SELECT.connect(self.on_cluster_select)

    def setup_detail_level_selector(self):
        self.detail_level_box = widgets.QComboBox(self)
        self.detail_level_box.addItem("Normal Detail", 1)
        self.detail_level_box.addItem("High Detail", 2)
        self.detail_level_box.addItem("All Waveforms", None)
        self.detail_level_box.activated.connect(self.update_detail_level)
        self.detail_level_box.setMaximumWidth(100)

        if self.dataset.count <= 30000:
            # The entire dataset does not have many waveforms
            self.set_detail_level(2)
            self.detail_level_box.setCurrentIndex(2)
        elif len(self.dataset.flatten(1)) <= 500:
            # The fully clustered dataset is very small
            self.set_detail_level(1)
            self.detail_level_box.setCurrentIndex(1)
        else:
            self.set_detail_level(0)
            self.detail_level_box.setCurrentIndex(0)

    def update_detail_level(self, index):
        self.set_detail_level(index)
        self.reset()

    def set_detail_level(self, index):
        self.flatten_level = self.detail_level_box.itemData(index)

    def reset(self):
        # Delete the old layout
        # widgets.QWidget().setLayout(self.layout())
        for label, scatters in self.scatters.items():
            for scat in scatters:
                scat.remove()
            del scatters[:]
        for scat in self.main_scatters:
            scat.remove()
        del self.main_scatters[:]
        self.setup_data()

    @property
    def dataset(self):
        return self.parent().dataset

    @property
    def colors(self):
        return self.parent().colors

    @property
    def selected(self):
        return self.parent().selected

    def setup_plots(self):
        fig = Figure()
        fig.patch.set_alpha(0.0)
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background-color:transparent;")

        self.axes = []
        for dim in range(self.ndim):
            ax = fig.add_axes(
                [0, 0.2 + (0.8 / self.ndim) * dim, 1, 0.8 / self.ndim],
                facecolor="#111111"
            )
            ax.set_yticks([])
            ax.patch.set_alpha(0.8)
            self.axes.append(ax)

        self.axes[0].xaxis.set_tick_params(rotation=45, labelsize=8)

        for ax in self.axes[1:]:
            ax.set_yticks([])
            ax.set_xticks([])

        self.scatters = defaultdict(list)
        self.main_scatters = []

        self._frame = 0
        self.parent().animation_timer.timeout.connect(self.rotate)

    def rotate(self):
        if not len(self.dataset.nodes) or not self.main_scatters:
            return

        _t = (
                (2 * np.pi) *
                (self._frame % self._rotation_period) /
                self._rotation_period
        )
        for dim in range(self.ndim):
            self.main_scatters[dim].set_offsets(np.array([
                self.flattened.times,
                (
                    np.cos(_t + dim * np.pi / 2) * self.pcs.T[dim] +
                    np.sin(_t + dim * np.pi / 2) * self.pcs.T[dim + 1]
                )
            ]).T)
        for label, node in zip(self.dataset.labels, self.dataset.nodes):
            pcs = self.pcs[self.flattened.labels == label]
            for dim in range(self.ndim):
                self.scatters[label][dim].set_offsets(np.array([
                    self.flattened.times[self.flattened.labels == label],
                    (
                        np.cos(_t + dim * np.pi / 2) * pcs.T[dim] +
                        np.sin(_t + dim * np.pi / 2) * pcs.T[dim + 1]
                    )
                ]).T)
        self._frame += 1
        self._frame = self._frame % self._rotation_period
        self.canvas.draw_idle()

    def setup_data(self):
        if not len(self.dataset.nodes):
            self.canvas.draw_idle()
            return

        self.scatters = defaultdict(list)
        self.main_scatters = []

        self.flattened = self.dataset.flatten(self.flatten_level)
        skip = max(1, len(self.flattened) // self.max_points)
        self.flattened = self.flattened.select(slice(None, None, skip))

        self.pca = PCA(n_components=self.ndim + 1, whiten=True).fit(self.flattened.waveforms)
        self.pcs = self.pca.transform(self.flattened.waveforms)

        s = self.detail_level_scatter_size[self.flatten_level]
        for dim in range(self.ndim):
            self.main_scatters.append(self.axes[dim].scatter(
                self.flattened.times,
                self.pcs.T[dim],
                s=s,
                alpha=0.8,
                color="Gray",
                rasterized=True
            ))

        for label, node in zip(self.dataset.labels, self.dataset.nodes):
            pcs = self.pcs[self.flattened.labels == label]
            times = self.flattened.times[self.flattened.labels == label]
            for dim in range(self.ndim):
                self.scatters[label].append(
                    self.axes[dim].scatter(
                        times,
                        pcs.T[dim],
                        s=3 * s,
                        alpha=1,
                        color=self.colors[label],
                        rasterized=True
                    )
                )
                self.scatters[label][-1].set_visible(label in self.selected)

        for dim, ax in enumerate(self.axes):
            ylim = np.max(np.abs(self.pcs[:, dim])) * 1.2
            ax.set_ylim(-ylim, ylim)
            fully_flat = self.dataset.flatten().times
            xlim = (fully_flat[0], fully_flat[-1])
            ax.set_xlim(*xlim)

        self.canvas.draw_idle()

    def on_cluster_select(self, selected, old_selected):
        for label in self.scatters:
            for scat in self.scatters[label]:
                # TODO (kevin): i dont think this set color is needed
                # scat.set_color(self.colors[label])
                scat.set_visible(label in selected)

    def on_cluster_highlight(self, new_highlight, old_highlight, temporary):
        if (
                old_highlight is not None and
                old_highlight in self.scatters and
                old_highlight in self.colors
                ):
            for scat in self.scatters[old_highlight]:
                scat.set_color(self.colors[old_highlight])
                scat.set_visible(old_highlight in self.selected)

        if new_highlight is not None:
            for scat in self.scatters[new_highlight]:
                scat.set_facecolor("White")
                scat.set_edgecolor(self.colors[new_highlight])
                scat.set_visible(True)

    def init_ui(self):
        layout = widgets.QVBoxLayout()

        layout.addWidget(self.canvas)
        layout.addWidget(self.detail_level_box)

        self.setLayout(layout)
