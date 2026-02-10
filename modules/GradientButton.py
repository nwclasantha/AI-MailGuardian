import sys
import math

# CustomTkinter for modern UI
try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False
    print("Error: customtkinter is required. Install with: pip install customtkinter")
    sys.exit(1)

# Color scheme (only the colors used by GradientButton)
COLORS = {
    'accent_primary': '#00d4ff',
    'accent_secondary': '#00ff88',
}

class GradientButton(ctk.CTkButton):
    """Advanced button with gradient animation"""

    def __init__(self, *args, **kwargs):
        self.gradient_colors = kwargs.pop('gradient_colors', [COLORS['accent_primary'], COLORS['accent_secondary']])
        super().__init__(*args, **kwargs)
        self.bind("<Enter>", self.on_hover)
        self.bind("<Leave>", self.on_leave)
        self.animation_id = None

    def _is_disabled(self):
        """Check if button is currently disabled"""
        try:
            return str(self.cget("state")) == "disabled"
        except Exception:
            return False

    def on_hover(self, event):
        if self._is_disabled():
            return
        self.animate_gradient()

    def on_leave(self, event):
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None
        if not self._is_disabled():
            self.configure(fg_color=self.gradient_colors[0])

    def animate_gradient(self, step=0):
        try:
            if not self.winfo_exists():
                return
            if self._is_disabled():
                self.animation_id = None
                return

            progress = (math.sin(step * 0.1) + 1) / 2
            r1, g1, b1 = self.hex_to_rgb(self.gradient_colors[0])
            r2, g2, b2 = self.hex_to_rgb(self.gradient_colors[1])

            r = int(r1 + (r2 - r1) * progress)
            g = int(g1 + (g2 - g1) * progress)
            b = int(b1 + (b2 - b1) * progress)

            color = f"#{r:02x}{g:02x}{b:02x}"
            self.configure(fg_color=color)

            self.animation_id = self.after(50, lambda: self.animate_gradient(step + 1))
        except Exception:
            if self.winfo_exists() and hasattr(self, 'gradient_colors') and self.gradient_colors:
                try:
                    self.configure(fg_color=self.gradient_colors[0])
                except Exception:
                    pass

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
