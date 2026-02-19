import sys
import warnings

warnings.filterwarnings('ignore')

# GUI Framework
import tkinter as tk

# CustomTkinter for modern UI
try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False
    print("Error: customtkinter is required. Install with: pip install customtkinter")
    sys.exit(1)

# Color constants used by this component — Modern Dark Pro
COLORS = {
    'bg_primary': '#0f1117',
    'bg_secondary': '#161b22',
    'bg_tertiary': '#1c2128',
    'bg_quaternary': '#272d37',
    'accent_primary': '#4f8ff7',
    'accent_secondary': '#3fb950',
    'accent_tertiary': '#a78bfa',
    'danger': '#f85149',
    'warning': '#d29922',
    'success': '#3fb950',
    'info': '#58a6ff',
    'text_primary': '#e6edf3',
    'text_secondary': '#8b949e',
    'text_tertiary': '#6e7681',
    'gradient_start': '#4f8ff7',
    'gradient_mid': '#a78bfa',
    'gradient_end': '#3fb950',
    'glass': 'rgba(255,255,255,0.03)',
    'shadow': 'rgba(0,0,0,0.4)',
    'highlight': '#f0f6fc',
    'border': '#30363d',
}

class AnimatedProgressBar(ctk.CTkFrame):
    """Advanced progress bar with wave animation"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(fg_color="transparent")

        self.canvas = tk.Canvas(self, height=30, bg='#161b22', highlightthickness=0)
        self.canvas.pack(fill="x", expand=True)

        self.progress = 0
        self._after_id = None
        # Pre-compute gradient endpoints (constant, no need to compute per-pixel)
        self._rgb_start = self.hex_to_rgb('#4f8ff7')
        self._rgb_end = self.hex_to_rgb('#3fb950')
        self.animate()

    def set_progress(self, value):
        self.progress = max(0, min(1, value))
        # animate() loop redraws every 50ms — no forced redraw needed

    def animate(self):
        try:
            if not self.winfo_exists():
                return
            if not self.canvas.winfo_exists():
                return

            self.canvas.delete("all")
            width = self.canvas.winfo_width()
            height = 30

            if width <= 1:
                width = self.winfo_width()
            if width <= 1:
                width = 400

            self.canvas.create_rectangle(0, 0, width, height, fill='#1c2128', outline="")

            progress_width = int(width * self.progress)
            if progress_width > 0:
                r1, g1, b1 = self._rgb_start
                r2, g2, b2 = self._rgb_end
                # Use ~20 segments instead of per-pixel to reduce canvas items
                num_segments = min(20, progress_width)
                seg_width = progress_width / num_segments
                for s in range(num_segments):
                    x_start = int(s * seg_width)
                    x_end = int((s + 1) * seg_width)
                    color_progress = (s + 0.5) / num_segments

                    r = int(r1 + (r2 - r1) * color_progress)
                    g = int(g1 + (g2 - g1) * color_progress)
                    b = int(b1 + (b2 - b1) * color_progress)

                    color = f"#{r:02x}{g:02x}{b:02x}"
                    self.canvas.create_rectangle(x_start, 0, x_end, height, fill=color, outline="")

            text = f"{int(self.progress * 100)}%"
            self.canvas.create_text(width / 2, height / 2, text=text, fill='#e6edf3',
                                    font=('Segoe UI', 11, 'bold'))

            self._after_id = self.after(50, self.animate)
        except tk.TclError:
            self._after_id = None  # Widget destroyed — do not reschedule
        except Exception:
            if self.winfo_exists() and self.canvas.winfo_exists():
                self._after_id = self.after(100, self.animate)
            else:
                self._after_id = None

    def destroy(self):
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        super().destroy()

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
