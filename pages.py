import tkinter as tk
from tkinter import ttk, filedialog
from data_loaders import LoadBatchFromDirectory
from clusterer import Clusterer
import threading
import umap
import pandas as pd


LARGEFONT = ("Verdana", 35)


class BasePage(tk.Frame):
    def __init__(self, parent, controller, title, message, back_button_link, continue_button_link):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller

    def add_title(self, title):
        page_title = ttk.Label(self, text=title, font=LARGEFONT)
        page_title.grid(row=0, column=4, padx=10, pady=10)

    def add_message(self, message):
        message = ttk.Label(self, text=message)
        message.grid(row=1, column=4, padx=10, pady=10)

    def add_back_button(self, link):
        about_button = ttk.Button(self, text="back", command=lambda: self.controller.show_frame(link))
        about_button.grid(row=1, column=1, padx=10, pady=10)

    def add_continue_button(self, link):
        start_button = ttk.Button(self, text="continue", command=lambda: self.controller.show_frame(link))
        start_button.grid(row=2, column=1, padx=10, pady=10)


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        page_title = ttk.Label(self, text="Lazy Labeller", font=LARGEFONT)

        page_title.grid(row=0, column=4, padx=10, pady=10)

        about_button = ttk.Button(self, text="About",
                             command=lambda: controller.show_frame(AboutPage))

        about_button.grid(row=1, column=1, padx=10, pady=10)

        start_button = ttk.Button(self, text="Start",
                             command=lambda: controller.show_frame(SelectDataPage))

        start_button.grid(row=2, column=1, padx=10, pady=10)


# second window frame page1
class AboutPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        page_title = ttk.Label(self, text="About", font=LARGEFONT)
        page_title.grid(row=0, column=4, padx=10, pady=10)
        about_text = tk.Text(self, wrap=tk.WORD, width=40, height=10)
        about_text.insert("1.0",
                           "this app uses machine learning methods to identify groups of similar images in large " 
                           "training data sets, and presents them as batches for labelling via this GUI. " 
                           "an active learning approach is then leveraged to iteratively improve the quality and " 
                           "performance of the model, while minimising the amount of boring manual labelling")

        about_text.config(state=tk.DISABLED)

        about_text.grid(row=1, column=4, padx=10, pady=10)

        back_button = ttk.Button(self, text="back",
                             command=lambda: controller.show_frame(StartPage))

        back_button.grid(row=1, column=1, padx=10, pady=10)

        start_button = ttk.Button(self, text="start",
                             command=lambda: controller.show_frame(SelectDataPage))

        start_button.grid(row=2, column=1, padx=10, pady=10)


# third window frame page2
class SelectDataPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        page_title = ttk.Label(self, text="Select data", font=LARGEFONT)
        page_title.grid(row=0, column=4, padx=10, pady=10)

        back_button = ttk.Button(self, text="back", command=lambda: controller.show_frame(StartPage))

        back_button.grid(row=1, column=1, padx=10, pady=10)

        message = ttk.Label(self, text="Select a directory containing your data")
        message.grid(row=1, column=4, padx=10, pady=10)

        select_data_button = ttk.Button(self, text="select data directory", command=self.select_data)

        select_data_button.grid(row=2, column=4, padx=10, pady=10)

    def select_data(self):
        directory = filedialog.askdirectory(parent=self, initialdir='.', title='Please select a directory')
        self.pass_data(directory)
        self.controller.show_frame(ConfirmData)

    def pass_data(self, directory):
        self.controller.shared_data['directory'] = directory


class ConfirmData(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        page_title = ttk.Label(self, text="Confirm", font=LARGEFONT)
        page_title.grid(row=0, column=4, padx=10, pady=10)

        self.directory = controller.shared_data['directory']

        message = ttk.Label(self, text=f"please confirm directory {self.directory} contains your data")
        message.grid(row=1, column=4, padx=10, pady=10)

        back_button = ttk.Button(self, text="back", command=lambda: controller.show_frame(SelectDataPage))

        back_button.grid(row=1, column=1, padx=10, pady=10)

        continue_button = ttk.Button(self, text="continue", command=lambda: controller.show_frame(ProcessData))

        continue_button.grid(row=2, column=1, padx=10, pady=10)


class ProcessData(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        page_title = ttk.Label(self, text="Processing...", font=LARGEFONT)
        page_title.grid(row=0, column=4, padx=10, pady=10)
        back_button = ttk.Button(self, text="back", command=lambda: controller.show_frame(SelectDataPage))
        back_button.grid(row=1, column=1, padx=10, pady=10)
        continue_button = ttk.Button(self, text="continue", command=lambda: controller.show_frame(SelectDataPage))
        continue_button.grid(row=2, column=1, padx=10, pady=10)
        reading_message = ttk.Label(self, text="reading data...........[ ]")
        reading_message.grid(row=1, column=4, padx=10, pady=10)
        extracting_message = ttk.Label(self, text="extracting data........[ ]")
        extracting_message.grid(row=2, column=4, padx=10, pady=10)
        reducing_message = ttk.Label(self, text="reducing dimensions....[ ]")
        reducing_message.grid(row=3, column=4, padx=10, pady=10)
        clustering_message = ttk.Label(self, text="clustering data........[ ]")
        clustering_message.grid(row=4, column=4, padx=10, pady=10)
        if controller.batches is None:
            controller.batches = LoadBatchFromDirectory(directory=controller.shared_data['directory'], batch_size=2)
        batch = controller.batches.load_batch()
        print(batch)
        reading_message = ttk.Label(self, text="reading data...........[X]")
        reading_message.grid(row=1, column=4, padx=10, pady=10)
        features = controller.feature_extractor.get_features(batch)
        extracting_message = ttk.Label(self, text="extracting data........[X]")
        extracting_message.grid(row=2, column=4, padx=10, pady=10)

        x_list, y_list = controller.reducer.reduce_features(features)

        reducing_message = ttk.Label(self, text="reducing dimensions....[X]")
        reducing_message.grid(row=3, column=4, padx=10, pady=10)

        clust = Clusterer(x_list, y_list)

        clustering_message = ttk.Label(self, text="clustering data........[ ]")
        clustering_message.grid(row=4, column=4, padx=10, pady=10)

        cluster_labels = clust.labels

        df = pd.DataFrame(
            dict(x=x_list, y=y_list, cluster=cluster_labels, paths=paths))

        controller.shared_data['clustered_data'] = df

        # fig = px.scatter(df, x='x', y='y', symbol='labels', color='cluster', hover_data=['labels', 'paths'])
        # fig.show()
