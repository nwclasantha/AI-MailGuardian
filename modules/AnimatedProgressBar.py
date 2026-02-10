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

class AnimatedProgressBar(ctk.CTkFrame):
    """Advanced progress bar with wave animation"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(fg_color="transparent")

        self.canvas = tk.Canvas(self, height=30, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(fill="x", expand=True)

        self.progress = 0
        self.wave_offset = 0
        self.animate()

    def set_progress(self, value):
        self.progress = max(0, min(1, value))
        # Force immediate redraw when progress changes
        self.update_display()

    def update_display(self):
        """Immediately update the progress bar display"""
        try:
            self.canvas.update_idletasks()
            self.canvas.delete("all")
            width = self.canvas.winfo_width()
            height = 30

            if width <= 1:
                width = self.winfo_width()
            if width <= 1:
                width = 400  # fallback width

            if width > 1:
                # Draw background
                self.canvas.create_rectangle(0, 0, width, height, fill='#252525', outline="")

                # Draw progress bar
                progress_width = int(width * self.progress)
                if progress_width > 0:
                    for i in range(progress_width):
                        color_progress = i / max(progress_width, 1)
                        r1, g1, b1 = self.hex_to_rgb('#00d4ff')
                        r2, g2, b2 = self.hex_to_rgb('#00ff88')

                        r = int(r1 + (r2 - r1) * color_progress)
                        g = int(g1 + (g2 - g1) * color_progress)
                        b = int(b1 + (b2 - b1) * color_progress)

                        color = f"#{r:02x}{g:02x}{b:02x}"
                        self.canvas.create_rectangle(i, 0, i + 1, height, fill=color, outline="")

                # Draw percentage text
                text = f"{int(self.progress * 100)}%"
                self.canvas.create_text(width / 2, height / 2, text=text, fill='#ffffff',
                                        font=('Arial', 12, 'bold'))

            # Force canvas update
            self.canvas.update_idletasks()
        except Exception:
            pass

    def animate(self):
        try:
            if not self.winfo_exists():
                return

            self.canvas.delete("all")
            width = self.canvas.winfo_width()
            height = 30

            if width <= 1:
                width = self.winfo_width()
            if width <= 1:
                width = 400

            self.canvas.create_rectangle(0, 0, width, height, fill='#252525', outline="")

            progress_width = int(width * self.progress)
            if progress_width > 0:
                for i in range(progress_width):
                    color_progress = i / max(progress_width, 1)
                    r1, g1, b1 = self.hex_to_rgb('#00d4ff')
                    r2, g2, b2 = self.hex_to_rgb('#00ff88')

                    r = int(r1 + (r2 - r1) * color_progress)
                    g = int(g1 + (g2 - g1) * color_progress)
                    b = int(b1 + (b2 - b1) * color_progress)

                    color = f"#{r:02x}{g:02x}{b:02x}"
                    self.canvas.create_rectangle(i, 0, i + 1, height, fill=color, outline="")

            text = f"{int(self.progress * 100)}%"
            self.canvas.create_text(width / 2, height / 2, text=text, fill='#ffffff',
                                    font=('Arial', 12, 'bold'))

            self.wave_offset += 2
            self.after(50, self.animate)
        except Exception:
            if self.winfo_exists():
                self.after(100, self.animate)

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
