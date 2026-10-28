import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
import numpy as np
import threading
from pathlib import Path
import sys
import time
import csv
from collections import Counter
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from predict import SatellitePredictor


from gui.confusion_tools import ConfusionTools


class SatelliteGUI(ConfusionTools):
    """Main GUI application for satellite image classification."""
    
    def __init__(self, root, config_path="config.json"):
        self.root = root
        self.config_path = config_path
        self.predictor = None
        self.current_image_path = None
        self.current_image_tk = None
        self.history = []
        self.zoom = 1.0
        self.dark = False
        self.threshold = tk.DoubleVar(value=80.0)
        
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        self.install_confusion_tools()
        self.load_model_async()
    
    def setup_window(self):
        """Configure main window."""
        self.root.title(self.config_path and self.get_config_value('gui', 'window_title') 
                       or "Satellite Image Classification System")
        self.root.geometry(self.get_config_value('gui', 'window_size') or "900x700")
        self.root.minsize(800, 600)
        
        try:
            self.root.iconbitmap(default='')
        except:
            pass
    
    def get_config_value(self, section, key):
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config.get(section, {}).get(key)
        except:
            return None
    
    def setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'), foreground='#2c3e50')
        style.configure('Heading.TLabel', font=('Segoe UI', 12, 'bold'), foreground='#34495e')
        style.configure('Result.TLabel', font=('Segoe UI', 11), foreground='#2c3e50')
        style.configure('Confidence.TLabel', font=('Segoe UI', 14, 'bold'), foreground='#27ae60')
        style.configure('Error.TLabel', font=('Segoe UI', 11), foreground='#e74c3c')
        style.configure('Custom.TButton', font=('Segoe UI', 10), padding=10)
        style.configure('Big.TButton', font=('Segoe UI', 11, 'bold'), padding=15)
    
    def create_widgets(self):
        """Create all GUI widgets."""
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.create_header(main_frame)
        self.create_content(main_frame)
        self.create_extras(main_frame)
        self.create_footer(main_frame)
    
    def create_header(self, parent):
        """Create header section."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title = ttk.Label(header_frame, text="🛰 Satellite Image Classification", style='Title.TLabel')
        title.pack()
        
        subtitle = ttk.Label(header_frame, 
                           text="Upload a satellite image to classify land use / land cover",
                           font=('Segoe UI', 10), foreground='#7f8c8d')
        subtitle.pack(pady=(5, 0))
        
        separator = ttk.Separator(header_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=15)
    
    def create_content(self, parent):
        """Create main content area."""
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        self.create_image_area(left_panel)
        self.create_results_area(right_panel)
        self.create_batch_area(right_panel)
    
    def create_image_area(self, parent):
        """Create image display and upload area."""
        img_frame = ttk.LabelFrame(parent, text=" Satellite Image ", padding=15)
        img_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_canvas = tk.Canvas(img_frame, bg='#ecf0f1', highlightthickness=1, 
                                      highlightbackground='#bdc3c7')
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(self.image_canvas, text="No image loaded\n\nClick 'Upload Image' to select a satellite image",
                                    font=('Segoe UI', 11), foreground='#95a5a6', background='#ecf0f1')
        self.image_label.place(relx=0.5, rely=0.5, anchor='center')
        
        btn_frame = ttk.Frame(img_frame)
        btn_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.upload_btn = ttk.Button(btn_frame, text="📁 Upload Image", style='Big.TButton',
                                    command=self.upload_image)
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.predict_btn = ttk.Button(btn_frame, text="🔍 Classify", style='Big.TButton',
                                     command=self.predict_image, state=tk.DISABLED)
        self.predict_btn.pack(side=tk.LEFT)
        
        self.clear_btn = ttk.Button(btn_frame, text="🗑 Clear", style='Custom.TButton',
                                   command=self.clear_image, state=tk.DISABLED)
        self.clear_btn.pack(side=tk.RIGHT)
    
    def create_results_area(self, parent):
        """Create prediction results area."""
        results_frame = ttk.LabelFrame(parent, text=" Prediction Results ", padding=15)
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.result_class = ttk.Label(results_frame, text="--", style='Result.TLabel')
        self.result_class.pack(anchor='w')
        
        self.result_confidence = ttk.Label(results_frame, text="--", style='Confidence.TLabel')
        self.result_confidence.pack(anchor='w', pady=(5, 0))
        
        separator = ttk.Separator(results_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=15)
        
        ttk.Label(results_frame, text="All Class Probabilities:", style='Heading.TLabel').pack(anchor='w')
        
        self.prob_frame = ttk.Frame(results_frame)
        self.prob_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.prob_bars = {}
        self.prob_labels = {}
        
        for cls in self.get_config_value('dataset', 'class_names') or []:
            row = ttk.Frame(self.prob_frame)
            row.pack(fill=tk.X, pady=2)
            
            lbl = ttk.Label(row, text=f"{cls}:", width=20, anchor='w', font=('Segoe UI', 9))
            lbl.pack(side=tk.LEFT)
            
            bar = ttk.Progressbar(row, length=150, mode='determinate')
            bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            val_lbl = ttk.Label(row, text="0.0%", width=8, anchor='e', font=('Segoe UI', 9))
            val_lbl.pack(side=tk.LEFT, padx=(5, 0))
            
            self.prob_bars[cls] = bar
            self.prob_labels[cls] = val_lbl
    
    def create_batch_area(self, parent):
        """Create batch processing area."""
        batch_frame = ttk.LabelFrame(parent, text=" Batch Processing ", padding=15)
        batch_frame.pack(fill=tk.X)
        
        ttk.Label(batch_frame, text="Process multiple images at once:", 
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(0, 5))
        
        btn_frame = ttk.Frame(batch_frame)
        btn_frame.pack(fill=tk.X)
        
        self.batch_btn = ttk.Button(btn_frame, text="📂 Select Folder", style='Custom.TButton',
                                   command=self.batch_process)
        self.batch_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.export_btn = ttk.Button(btn_frame, text="💾 Export Results", style='Custom.TButton',
                                    command=self.export_results, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT)
        
        self.batch_results = []
        self.batch_tree = None
    
    def create_footer(self, parent):
        """Create footer with status bar."""
        footer_frame = ttk.Frame(parent)
        footer_frame.pack(fill=tk.X, pady=(20, 0))
        
        separator = ttk.Separator(footer_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=(0, 10))
        
        self.status_var = tk.StringVar(value="Loading model...")
        status_label = ttk.Label(footer_frame, textvariable=self.status_var, 
                                font=('Segoe UI', 9), foreground='#7f8c8d')
        status_label.pack(side=tk.LEFT)
        
        self.model_status = ttk.Label(footer_frame, text="Model: Not Loaded", 
                                     font=('Segoe UI', 9), foreground='#e74c3c')
        self.model_status.pack(side=tk.RIGHT)
    
    def load_model_async(self):
        """Load model in background thread."""
        def load():
            try:
                self.predictor = SatellitePredictor(self.config_path)
                self.predictor.load_model()
                self.root.after(0, self.on_model_loaded)
            except Exception as e:
                self.root.after(0, lambda: self.on_model_error(str(e)))
        
        threading.Thread(target=load, daemon=True).start()
    
    def on_model_loaded(self):
        self.status_var.set("Ready - Upload an image to classify")
        self.model_status.config(text="Model: Loaded ✓", foreground='#27ae60')
        self.upload_btn.config(state=tk.NORMAL)
    
    def on_model_error(self, error):
        self.status_var.set("Error loading model")
        self.model_status.config(text="Model: Failed ✗", foreground='#e74c3c')
        messagebox.showerror("Model Load Error", 
                           f"Could not load trained model:\n{error}\n\nPlease train the model first using train.py")
    
    def upload_image(self):
        """Handle image upload."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
            ("All files", "*.*")
        ]
        
        path = filedialog.askopenfilename(title="Select Satellite Image", filetypes=filetypes)
        
        if path:
            self.load_image(path)
    
    def load_image(self, path):
        self.cm_single = []
        """Load and display image."""
        try:
            self.current_image_path = path
            
            img = Image.open(path)
            img.thumbnail((500, 400), Image.Resampling.LANCZOS)
            
            self.current_image_tk = ImageTk.PhotoImage(img)
            
            self.image_canvas.delete("all")
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width <= 1:
                canvas_width = 500
            if canvas_height <= 1:
                canvas_height = 400
            
            x = canvas_width // 2
            y = canvas_height // 2
            
            self.image_canvas.create_image(x, y, image=self.current_image_tk, anchor='center')
            self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))
            
            self.image_label.place_forget()
            
            self.predict_btn.config(state=tk.NORMAL)
            self.clear_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Loaded: {Path(path).name}")
            
            self.clear_results()
            
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Could not load image:\n{e}")
    
    def predict_image(self):
        """Run prediction on current image."""
        if not self.current_image_path or not self.predictor:
            return
        
        self.status_var.set("Classifying...")
        self.predict_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        
        def predict():
            try:
                start = time.perf_counter()
                result = self.predictor.predict_single(self.current_image_path)
                result["processing_seconds"] = time.perf_counter() - start
                result["path"] = self.current_image_path
                self.root.after(0, lambda: self.display_results(result))
            except Exception as e:
                self.root.after(0, lambda: self.on_prediction_error(str(e)))
        
        threading.Thread(target=predict, daemon=True).start()
    
    def display_results(self, result):
        """Display prediction results."""
        self.cm_single = [dict(result)]
        self.record_result(result)
        self.result_class.config(text=f"Predicted Class: {result['predicted_class']}")
        self.result_confidence.config(text=f"Confidence: {result['confidence']:.2f}%")
        
        for cls, prob in result['all_probabilities'].items():
            if cls in self.prob_bars:
                self.prob_bars[cls]['value'] = prob
                self.prob_labels[cls].config(text=f"{prob:.1f}%")
        
        self.predict_btn.config(state=tk.NORMAL)
        self.upload_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Classification complete in {result.get('processing_seconds', 0):.2f}s: {result['predicted_class']}")
    
    def on_prediction_error(self, error):
        messagebox.showerror("Prediction Error", f"Could not classify image:\n{error}")
        self.predict_btn.config(state=tk.NORMAL)
        self.upload_btn.config(state=tk.NORMAL)
        self.status_var.set("Prediction failed")
    
    def clear_results(self):
        """Clear prediction results."""
        self.result_class.config(text="--")
        self.result_confidence.config(text="--")
        for cls in self.prob_bars:
            self.prob_bars[cls]['value'] = 0
            self.prob_labels[cls].config(text="0.0%")
    
    def clear_image(self):
        """Clear current image."""
        self.current_image_path = None
        self.current_image_tk = None
        
        self.image_canvas.delete("all")
        self.image_label.place(relx=0.5, rely=0.5, anchor='center')
        
        self.predict_btn.config(state=tk.DISABLED)
        self.clear_btn.config(state=tk.DISABLED)
        
        self.clear_results()
        self.status_var.set("Ready - Upload an image to classify")
    
    def batch_process(self):
        """Process multiple images from a folder."""
        folder = filedialog.askdirectory(title="Select Folder with Satellite Images")
        
        if not folder:
            return
        
        supported = self.get_config_value('gui', 'supported_formats') or ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        # Scan once: extension matching is case-insensitive, so each file
        # is processed exactly once even when formats overlap.
        extensions = {str(ext).lower() if str(ext).startswith('.') else '.' + str(ext).lower()
                      for ext in supported}
        image_files = sorted(
            (p for p in Path(folder).iterdir()
             if p.is_file() and p.suffix.lower() in extensions),
            key=lambda p: p.name.lower()
        )
        
        if not image_files:
            messagebox.showinfo("No Images", "No supported image files found in the selected folder.")
            return
        
        self.status_var.set(f"Processing {len(image_files)} images...")
        self.batch_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        
        def process():
            try:
                results = []
                for index, f in enumerate(image_files, 1):
                    started = time.perf_counter()
                    try:
                        r = self.predictor.predict_single(str(f))
                        r = {k: v for k, v in r.items() if k not in ("original_image", "processed_image")}
                        r["path"] = str(f)
                        r["processing_seconds"] = time.perf_counter() - started
                    except Exception as exc:
                        r = {"path": str(f), "error": str(exc)}
                    results.append(r)
                    self.root.after(0, lambda n=index, total=len(image_files): self.update_progress(n, total))
                self.root.after(0, lambda: self.show_batch_results(results))
            except Exception as e:
                self.root.after(0, lambda: self.on_batch_error(str(e)))
        
        threading.Thread(target=process, daemon=True).start()
    
    def show_batch_results(self, results):
        """Show batch results in a new window."""
        batch_window = tk.Toplevel(self.root)
        batch_window.title("Batch Processing Results")
        batch_window.geometry("800x600")
        
        self.batch_results = results
        for r in results:
            if "error" not in r:
                self.record_result(r)
        
        columns = ('Filename', 'Predicted Class', 'Confidence', 'Status', 'Seconds')
        tree = ttk.Treeview(batch_window, columns=columns, show='headings', height=20)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=200 if col == 'Filename' else 150, anchor='center')
        
        for r in results:
            if 'error' not in r:
                filename = Path(r['path']).name
                tree.insert('', 'end', values=(filename, r['predicted_class'], f"{r['confidence']:.2f}%", self.confidence_status(r['confidence']), f"{r.get('processing_seconds', 0):.2f}"))
            else:
                tree.insert('', 'end', values=(Path(r['path']).name, f"ERROR: {r['error']}", "--"))
        
        scrollbar = ttk.Scrollbar(batch_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
        btn_frame = ttk.Frame(batch_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(btn_frame, text="Export to CSV", command=lambda: self.export_batch_csv(results)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Excel", command=lambda: self.export_excel(results)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="PDF Report", command=lambda: self.export_pdf(results)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Close", command=batch_window.destroy).pack(side=tk.RIGHT)
        
        self.batch_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Batch processing complete: {len(results)} images")
    
    def on_batch_error(self, error):
        messagebox.showerror("Batch Error", f"Batch processing failed:\n{error}")
        self.batch_btn.config(state=tk.NORMAL)
        self.status_var.set("Batch processing failed")
    
    def export_batch_csv(self, results):
        """Export batch results to CSV."""
        import csv
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                           filetypes=[("CSV files", "*.csv")],
                                           title="Save Batch Results")
        if path:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Filename', 'Predicted Class', 'Confidence (%)'] + 
                               [f'Prob_{cls}' for cls in self.predictor.class_names])
                for r in results:
                    if 'error' not in r:
                        row = [Path(r['path']).name, r['predicted_class'], f"{r['confidence']:.2f}"]
                        for cls in self.predictor.class_names:
                            row.append(f"{r['all_probabilities'].get(cls, 0):.2f}")
                        writer.writerow(row)
            messagebox.showinfo("Export Complete", f"Results saved to {path}")
    
    def export_results(self):
        """Export latest prediction or batch results."""
        if self.batch_results:
            self.export_batch_csv(self.batch_results)
        elif self.history:
            self.export_batch_csv([self.history[-1]])
        else:
            messagebox.showinfo("No Results", "Classify an image first.")



    def confidence_status(self, value):
        return "HIGH" if value >= self.threshold.get() else ("MODERATE" if value >= 60 else "LOW - REVIEW")

    def record_result(self, result):
        entry = {k: v for k, v in result.items() if k not in ("original_image", "processed_image")}
        entry["timestamp"] = datetime.now().isoformat(timespec="seconds")
        self.history.append(entry)
        try:
            with open("prediction_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=float) + "\n")
        except OSError:
            pass

    def update_progress(self, n, total):
        self.progress["value"] = n * 100 / total
        self.status_var.set(f"Processing {n}/{total} images")

    def create_extras(self, parent):
        panel = ttk.LabelFrame(parent, text=" Analysis & Tools ", padding=6)
        panel.pack(fill=tk.X)
        row = ttk.Frame(panel)
        row.pack(fill=tk.X)
        for label, action in [
            ("Dashboard / Charts", self.dashboard), ("History", self.show_history),
            ("Compare Images", self.compare_images), ("Image Info", self.image_info),
            ("Zoom +", lambda: self.zoom_image(1.25)), ("Zoom -", lambda: self.zoom_image(0.8)),
            ("Theme", self.toggle_theme), ("PDF", lambda: self.export_pdf(self.batch_results or self.history[-1:])),
            ("Excel", lambda: self.export_excel(self.batch_results or self.history[-1:]))]:
            ttk.Button(row, text=label, command=action).pack(side=tk.LEFT, padx=2)
        ttk.Label(panel, text="High confidence threshold (%)").pack(side=tk.LEFT)
        ttk.Spinbox(panel, from_=50, to=100, increment=5, textvariable=self.threshold, width=5).pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(panel, maximum=100)
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=12)
        self.root.bind("<Control-o>", lambda e: self.upload_image())
        # Optional native drop target: install tkinterdnd2 for drag-and-drop.
        try:
            from tkinterdnd2 import DND_FILES
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', lambda e: self.load_image(self.root.tk.splitlist(e.data)[0]))
        except (ImportError, AttributeError, tk.TclError):
            pass

    def zoom_image(self, factor):
        if not self.current_image_path:
            return
        self.zoom = max(0.25, min(5.0, self.zoom * factor))
        with Image.open(self.current_image_path) as source:
            img = source.copy()
        img.thumbnail((int(500*self.zoom), int(400*self.zoom)))
        self.current_image_tk = ImageTk.PhotoImage(img)
        self.image_canvas.delete('all')
        self.image_canvas.create_image(max(250, self.image_canvas.winfo_width()//2),
                                       max(200, self.image_canvas.winfo_height()//2),
                                       image=self.current_image_tk)

    def toggle_theme(self):
        self.dark = not self.dark
        bg, fg = ('#202936', '#f4f6fb') if self.dark else ('#f0f0f0', '#202936')
        style = ttk.Style()
        style.configure('TFrame', background=bg)
        style.configure('TLabel', background=bg, foreground=fg)
        style.configure('TLabelframe', background=bg, foreground=fg)
        style.configure('TLabelframe.Label', background=bg, foreground=fg)
        self.root.configure(bg=bg)

    def dashboard(self):
        results = [r for r in self.history if 'error' not in r]
        if not results:
            messagebox.showinfo('Dashboard', 'Classify images first.')
            return
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        win = tk.Toplevel(self.root)
        win.title('Classification Analytics Dashboard')
        win.geometry('850x630')
        counts = Counter(r['predicted_class'] for r in results)
        avg = sum(float(r['confidence']) for r in results)/len(results)
        ttk.Label(win, text=f'Total: {len(results)}    Average confidence: {avg:.2f}%    Classes: {len(counts)}',
                  font=('Segoe UI', 13, 'bold')).pack(pady=12)
        fig = Figure(figsize=(9, 5), dpi=90)
        ax1, ax2 = fig.subplots(1, 2)
        ax1.bar(list(counts), list(counts.values()))
        ax1.tick_params(axis='x', labelrotation=75, labelsize=8)
        ax1.set_title('Predicted class counts')
        ax2.pie(list(counts.values()), labels=list(counts), autopct='%1.0f%%', textprops={'fontsize':7})
        ax2.set_title('Class distribution')
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_history(self):
        win = tk.Toplevel(self.root)
        win.title('Prediction History')
        win.geometry('850x500')
        columns = ('Time', 'Image', 'Class', 'Confidence', 'Status')
        tree = ttk.Treeview(win, columns=columns, show='headings')
        for c in columns:
            tree.heading(c, text=c)
            tree.column(c, width=155)
        for r in reversed(self.history):
            tree.insert('', 'end', values=(r.get('timestamp',''), Path(r.get('path','')).name,
                       r['predicted_class'], f"{r['confidence']:.2f}%", self.confidence_status(r['confidence'])))
        tree.pack(fill=tk.BOTH, expand=True)

    def compare_images(self):
        paths = filedialog.askopenfilenames(title='Select two satellite images', filetypes=[('Images','*.jpg *.jpeg *.png *.tif *.tiff')])
        if len(paths) != 2:
            messagebox.showinfo('Comparison', 'Select exactly two images.')
            return
        win = tk.Toplevel(self.root)
        win.title('Side-by-Side Satellite Image Comparison')
        win.geometry('850x480')
        win.photos = []
        self.cm_comparison = []
        for path in paths:
            col = ttk.Frame(win, padding=12)
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            with Image.open(path) as source:
                img = source.copy()
            img.thumbnail((390, 330))
            photo = ImageTk.PhotoImage(img)
            win.photos.append(photo)
            ttk.Label(col, image=photo).pack()
            ttk.Label(col, text=Path(path).name).pack()
            try:
                result = self.predictor.predict_single(path)
                self.cm_comparison.append({"path": path, "predicted_class": result["predicted_class"]})
                ttk.Label(col, text=f"{result['predicted_class']} ({result['confidence']:.2f}%)").pack()
            except Exception as exc:
                self.cm_comparison.append({'path': path, 'error': str(exc)})
                ttk.Label(col, text=f'Error: {exc}').pack()

    def image_info(self):
        if not self.current_image_path:
            messagebox.showinfo('Image Information', 'Select an image first.')
            return
        with Image.open(self.current_image_path) as img:
            info = f"File: {Path(self.current_image_path).name}\nDimensions: {img.width} x {img.height}\nFormat: {img.format}"
        messagebox.showinfo('Satellite Image Details', info + '\n\nClassification describes the predicted land-cover category, not verified ground truth.')

    def export_excel(self, results):
        if not results:
            messagebox.showinfo('Export', 'No results available.')
            return
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill
        except ImportError:
            messagebox.showerror('Missing Dependency', 'Install openpyxl to export Excel.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel', '*.xlsx')])
        if not path:
            return
        wb = Workbook()
        ws = wb.active
        ws.title = 'Predictions'
        ws.append(['Filename', 'Predicted Class', 'Confidence (%)', 'Status', 'Seconds', 'Timestamp'])
        for r in results:
            if 'error' not in r:
                ws.append([Path(r.get('path','')).name, r['predicted_class'], float(r['confidence']),
                           self.confidence_status(r['confidence']), float(r.get('processing_seconds',0)), r.get('timestamp','')])
        for cell in ws[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='24466B')
        for col, width in {'A':34,'B':25,'C':19,'D':23,'E':13,'F':24}.items():
            ws.column_dimensions[col].width = width
        ws.freeze_panes = 'A2'
        wb.save(path)
        messagebox.showinfo('Export', f'Saved {path}')

    def export_pdf(self, results):
        if not results:
            messagebox.showinfo('Report', 'No results available.')
            return
        try:
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4, landscape
        except ImportError:
            messagebox.showerror('Missing Dependency', 'Install reportlab to export PDF.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.pdf', filetypes=[('PDF', '*.pdf')])
        if not path:
            return
        styles = getSampleStyleSheet()
        story = [Paragraph('Satellite Classification Report', styles['Title']),
                 Paragraph(f'Generated: {datetime.now():%Y-%m-%d %H:%M}', styles['Normal']), Spacer(1, 15)]
        rows = [['Image', 'Predicted Class', 'Confidence', 'Status']]
        for r in results:
            if 'error' not in r:
                rows.append([Path(r.get('path','')).name[:35], r['predicted_class'],
                             f"{r['confidence']:.2f}%", self.confidence_status(r['confidence'])])
        for offset in range(1, len(rows), 32):
            table = Table([rows[0]] + rows[offset:offset+32], colWidths=[220,160,100,130], repeatRows=1)
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#24466B')),
                                      ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                                      ('GRID',(0,0),(-1,-1),0.3,colors.lightgrey),
                                      ('FONTSIZE',(0,0),(-1,-1),8)]))
            story.extend([table, Spacer(1, 14)])
        story.append(Paragraph('Confidence scores are model estimates, not verified accuracy.', styles['Normal']))
        SimpleDocTemplate(path, pagesize=landscape(A4)).build(story)
        messagebox.showinfo('Report', f'Saved {path}')

def main():
    root = tk.Tk()
    app = SatelliteGUI(root, "config.json")
    root.mainloop()


if __name__ == "__main__":
    main()