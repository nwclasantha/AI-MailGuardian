import sys
import numpy as np

# CustomTkinter for modern UI
try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False
    print("Error: customtkinter is required. Install with: pip install customtkinter")
    sys.exit(1)

# Matplotlib for charts
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RadarChart(ctk.CTkFrame):
    """Interactive radar chart for threat analysis"""

    def __init__(self, parent, categories, **kwargs):
        super().__init__(parent, **kwargs)
        self.categories = categories
        self.values = [0] * len(categories)
        self.configure(fg_color='#1a1a1a')

        self.fig = Figure(figsize=(6, 6), facecolor='#1a1a1a')
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_facecolor('#1a1a1a')

        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.update_chart()

    def update_values(self, values):
        # Ensure values length matches categories
        if len(values) != len(self.categories):
            if len(values) < len(self.categories):
                values = list(values) + [0] * (len(self.categories) - len(values))
            else:
                values = list(values[:len(self.categories)])
        self.values = values
        self.update_chart()

    def update_chart(self):
        self.ax.clear()

        if not self.categories or not self.values:
            self.ax.text(0.5, 0.5, 'No data', transform=self.ax.transAxes,
                        ha='center', va='center', color='#ffffff')
            self.canvas.draw()
            return

        angles = np.linspace(0, 2 * np.pi, len(self.categories), endpoint=False).tolist()
        values = self.values + [self.values[0]]
        angles += angles[:1]

        self.ax.plot(angles, values, color='#00d4ff', linewidth=2)
        self.ax.fill(angles, values, color='#00d4ff', alpha=0.25)

        self.ax.set_ylim(0, 100)
        self.ax.set_theta_offset(np.pi / 2)
        self.ax.set_theta_direction(-1)

        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(self.categories, color='#ffffff')
        self.ax.set_yticklabels([])

        self.ax.grid(True, color='#252525', linestyle='--', alpha=0.5)

        self.canvas.draw()
