import tkinter as tk

from botnet_detector.gui.app import BotnetDetectionApp


def main():
    root = tk.Tk()
    BotnetDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
