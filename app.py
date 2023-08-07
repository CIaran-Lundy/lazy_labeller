import tkinter as tk
from tkinter import ttk, filedialog

LARGEFONT = ("Verdana", 35)


class tkinterApp(tk.Tk):

    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)

        # creating a container
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # initializing frames to an empty array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, AboutPage, SelectDataPage):
            frame = F(container, self)

            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


# first window frame startpage

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
        page_title = ttk.Label(self, text="Select data", font=LARGEFONT)
        page_title.grid(row=0, column=4, padx=10, pady=10)

        back_button = ttk.Button(self, text="back",
                             command=lambda: controller.show_frame(StartPage))

        back_button.grid(row=1, column=1, padx=10, pady=10)

        page_title = ttk.Label(self, text="Select a directory containing your data")
        page_title.grid(row=1, column=4, padx=10, pady=10)

        file = filedialog.askdirectory(parent=self, initialdir='/', title='Please select a directory')



# Driver Code
app = tkinterApp()
app.mainloop()
