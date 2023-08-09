import tkinter as tk
from pages import *
from feature_extractors import FeatureExtractor
from dimension_reducers import UMAPReducer


class Controller:

    def __init__(self, *pages):
        self.batches = None
        self.feature_extractor = FeatureExtractor((100, 100, 3))
        self.reducer = UMAPReducer()
        self.frames = {}
        self.shared_data = {'directory': None}
        self.window = Window()
        self.show_frame(pages[0])

    def show_frame(self, frame):
        frame = frame(self.window.container, self)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()

    def start(self):
        self.window.mainloop()


class Window(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)


controller = Controller(StartPage, AboutPage, SelectDataPage, ProcessData)

controller.start()


