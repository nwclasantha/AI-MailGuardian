import sys
import random
import tkinter as tk

# CustomTkinter for modern UI
try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False
    print("Error: customtkinter is required. Install with: pip install customtkinter")
    sys.exit(1)

class ParticleEffect(ctk.CTkFrame):
    """Animated particle background effect"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(fg_color="transparent")

        self.canvas = tk.Canvas(self, bg='#0f1117', highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.particles = []
        self._after_id = None
        self.particle_colors = ['#4f8ff7', '#3fb950', '#a78bfa']
        self.create_particles()
        self.animate()

    def create_particles(self):
        width = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else 800
        height = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else 600
        for _ in range(50):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(1, 3)
            speed = random.uniform(0.5, 2)
            color = random.choice(self.particle_colors)
            self.particles.append({
                'x': x, 'y': y, 'size': size, 'speed': speed, 'color': color
            })

    def animate(self):
        try:
            if not self.winfo_exists():
                return

            self.canvas.delete("all")
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()

            if width > 1 and height > 1:
                for particle in self.particles:
                    x, y = particle['x'], particle['y']
                    size = particle['size']

                    glow_stipples = ['gray12', 'gray25', 'gray50']
                    for i in range(3):
                        glow_size = size + i * 2
                        self.canvas.create_oval(
                            x - glow_size, y - glow_size,
                            x + glow_size, y + glow_size,
                            fill=particle['color'], outline="", stipple=glow_stipples[i]
                        )

                    self.canvas.create_oval(
                        x - size, y - size,
                        x + size, y + size,
                        fill=particle['color'], outline=""
                    )

                    particle['y'] -= particle['speed']
                    if particle['y'] < -10:
                        particle['y'] = height + 10
                        particle['x'] = random.randint(0, max(width, 1))

            self._after_id = self.after(50, self.animate)
        except tk.TclError:
            self._after_id = None  # Widget destroyed — do not reschedule
        except Exception:
            if self.winfo_exists():
                self._after_id = self.after(100, self.animate)
            else:
                self._after_id = None

    def destroy(self):
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        super().destroy()
