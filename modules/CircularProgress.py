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

# Configure GUI theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Color constants used by this component
COLORS = {
    'bg_primary': '#0a0a0a',
    'bg_secondary': '#1a1a1a',
    'bg_tertiary': '#252525',
    'bg_quaternary': '#303030',
    'accent_primary': '#00d4ff',
    'accent_secondary': '#00ff88',
    'accent_tertiary': '#ff00ff',
    'danger': '#ff3366',
    'warning': '#ffaa00',
    'success': '#00ff88',
    'info': '#4466ff',
    'text_primary': '#ffffff',
    'text_secondary': '#b0b0b0',
    'text_tertiary': '#808080',
    'gradient_start': '#00d4ff',
    'gradient_mid': '#00ff88',
    'gradient_end': '#ff00ff',
    'glass': 'rgba(255,255,255,0.05)',
    'shadow': 'rgba(0,0,0,0.5)',
    'highlight': '#ffffff'
}


class CircularProgress(ctk.CTkFrame):
    """Circular progress indicator"""

    def __init__(self, parent, size=100, **kwargs):
        super().__init__(parent, **kwargs)
        self.size = size
        self.progress = 0

        canvas_bg = kwargs.get('canvas_bg', '#1a1a1a')
        self.canvas = tk.Canvas(self, width=size, height=size,
                                bg=canvas_bg, highlightthickness=0)
        self.canvas.pack()

        self.draw_progress()

    def draw_progress(self):
        try:
            self.canvas.delete("all")

            self.canvas.create_oval(10, 10, self.size - 10, self.size - 10,
                                    outline='#252525', width=8)

            extent = -360 * self.progress
            if extent != 0:
                self.canvas.create_arc(10, 10, self.size - 10, self.size - 10,
                                       start=90, extent=extent,
                                       outline='#00d4ff', width=8,
                                       style='arc')

            text = f"{int(self.progress * 100)}%"
            self.canvas.create_text(self.size / 2, self.size / 2, text=text,
                                    fill='#ffffff', font=('Arial', 16, 'bold'))

            self.canvas.update_idletasks()
        except Exception:
            pass

    def set_progress(self, value):
        self.progress = max(0, min(1, value))
        self.draw_progress()

    def pack_forget(self):
        super().pack_forget()

    def pack(self, **kwargs):
        super().pack(**kwargs)
