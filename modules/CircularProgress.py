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


class CircularProgress(ctk.CTkFrame):
    """Circular progress indicator"""

    def __init__(self, parent, size=100, **kwargs):
        canvas_bg = kwargs.pop('canvas_bg', '#161b22')
        super().__init__(parent, **kwargs)
        self.size = size
        self.progress = 0
        self.canvas = tk.Canvas(self, width=size, height=size,
                                bg=canvas_bg, highlightthickness=0)
        self.canvas.pack()

        self.draw_progress()

    def draw_progress(self):
        try:
            if not self.winfo_exists():
                return
            if not self.canvas.winfo_exists():
                return
            self.canvas.delete("all")

            self.canvas.create_oval(10, 10, self.size - 10, self.size - 10,
                                    outline='#30363d', width=8)

            extent = -360 * self.progress
            if extent != 0:
                self.canvas.create_arc(10, 10, self.size - 10, self.size - 10,
                                       start=90, extent=extent,
                                       outline='#4f8ff7', width=8,
                                       style='arc')

            text = f"{int(self.progress * 100)}%"
            self.canvas.create_text(self.size / 2, self.size / 2, text=text,
                                    fill='#e6edf3', font=('Segoe UI', 15, 'bold'))
        except Exception:
            pass

    def set_progress(self, value):
        self.progress = max(0, min(1, value))
        try:
            self.after(0, self.draw_progress)  # Schedule on main thread
        except Exception:
            pass

    def pack_forget(self):
        super().pack_forget()

    def pack(self, **kwargs):
        super().pack(**kwargs)
