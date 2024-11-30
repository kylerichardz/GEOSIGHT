from src.gui import GeoSightGUI
import tkinter as tk
from tkinter import messagebox

def main():
    try:
        app = GeoSightGUI()
        app.run()
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Startup Error",
            f"Failed to start GeoSight: {str(e)}\n\nPlease check your internet connection and try again."
        )
        root.destroy()

if __name__ == "__main__":
    main() 