import tkinter as tk
from tkinter import filedialog, messagebox
import json

class ParameterEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("PyBaMM Parameter Editor")
        self.entries = {}
        self.file_path = ""

        self.load_button = tk.Button(root, text="Load JSON", command=self.load_json)
        self.load_button.pack(pady=10)

        self.form_frame = tk.Frame(root)
        self.form_frame.pack()

        self.save_button = tk.Button(root, text="Save JSON", command=self.save_json)
        self.save_button.pack(pady=10)

    def load_json(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not self.file_path:
            return

        with open(self.file_path, "r") as f:
            self.params = json.load(f)

        # clear previous entries
        for widget in self.form_frame.winfo_children():
            widget.destroy()
        self.entries.clear()

        # create entry fields
        for i, (key, value) in enumerate(self.params.items()):
            tk.Label(self.form_frame, text=key).grid(row=i, column=0, sticky="w")
            entry = tk.Entry(self.form_frame, width=30)
            entry.insert(0, str(value))
            entry.grid(row=i, column=1)
            self.entries[key] = entry

    def save_json(self):
        if not self.file_path:
            messagebox.showerror("Error", "No file loaded.")
            return

        updated_params = {}
        for key, entry in self.entries.items():
            val = entry.get()
            try:
                updated_params[key] = float(val)
            except ValueError:
                updated_params[key] = val

        with open(self.file_path, "w") as f:
            json.dump(updated_params, f, indent=4)

        messagebox.showinfo("Saved", "Parameters updated successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ParameterEditor(root)
    root.mainloop()
