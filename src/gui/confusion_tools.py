"""Confusion matrices using explicit ground truth; no model changes."""
import csv
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np


def matrix_counts(records, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    index = {label: i for i, label in enumerate(labels)}
    skipped = 0
    for row in records:
        if row.get('error') or row.get('true_label') not in index or row.get('predicted_class') not in index:
            skipped += 1
            continue
        matrix[index[row['true_label']], index[row['predicted_class']]] += 1
    return matrix, skipped


def folder_label(path, labels):
    for parent in Path(path).parents:
        if parent.name in labels:
            return parent.name
    return ''


class ConfusionTools:
    def install_confusion_tools(self):
        self.cm_single = []
        self.cm_comparison = []
        menu = tk.Menu(self.root)
        tools = tk.Menu(menu, tearoff=False)
        tools.add_command(label='Individual image...', command=lambda: self.cm_open(self.cm_single, 'Individual image'))
        tools.add_command(label='Latest batch...', command=lambda: self.cm_open(getattr(self, 'batch_results', []), 'Batch'))
        tools.add_command(label='Latest comparison...', command=lambda: self.cm_open(self.cm_comparison, 'Comparison'))
        tools.add_separator()
        tools.add_command(label='Overall dataset folder...', command=self.cm_dataset)
        menu.add_cascade(label='Confusion Matrices', menu=tools)
        self.root.config(menu=menu)

    def cm_open(self, results, title):
        labels = self.get_config_value('dataset', 'class_names')
        rows = [dict(r) for r in results if not r.get('error')]
        failed = len(results) - len(rows)
        if not rows:
            messagebox.showinfo('Confusion matrix', 'Run the corresponding prediction first. No successful predictions are available.')
            return
        for r in rows:
            r.setdefault('true_label', folder_label(r.get('path', ''), labels))
        win = tk.Toplevel(self.root)
        win.title(title + ' — ground truth')
        win.geometry('780x480')
        ttk.Label(win, text='Supply known true labels. Folder-derived labels should be checked. Predictions are never used as true labels.', wraplength=740).pack(padx=12,pady=10)
        ttk.Label(win, text=f'{len(rows)} successful predictions; {failed} failed images excluded.').pack()
        tree = ttk.Treeview(win, columns=('path','predicted','truth'), show='headings', selectmode='extended')
        for col,text in [('path','Image path'),('predicted','Prediction'),('truth','True label')]:
            tree.heading(col,text=text)
        tree.pack(fill=tk.BOTH,expand=True,padx=10)
        def refresh():
            tree.delete(*tree.get_children())
            for i,r in enumerate(rows): tree.insert('', 'end', iid=str(i),values=(r.get('path',''),r['predicted_class'],r['true_label'] or 'UNLABELED'))
        refresh()
        controls=ttk.Frame(win);controls.pack(fill=tk.X,padx=10,pady=10)
        selected=tk.StringVar(value=labels[0])
        ttk.Combobox(controls,textvariable=selected,values=labels,state='readonly',width=22).pack(side=tk.LEFT)
        def assign():
            for item in tree.selection(): rows[int(item)]['true_label']=selected.get()
            refresh()
        ttk.Button(controls,text='Label selected rows',command=assign).pack(side=tk.LEFT,padx=5)
        def import_csv():
            path=filedialog.askopenfilename(parent=win,filetypes=[('Labels CSV','*.csv')])
            if not path:return
            try:
                with open(path,encoding='utf-8-sig',newline='') as f:
                    reader=csv.DictReader(f)
                    if not {'path','true_label'}.issubset(reader.fieldnames or []):raise ValueError('CSV needs path,true_label columns. Use the full image paths shown in the table.')
                    mapping={}
                    for entry in reader:
                        key=str(Path(entry['path']).resolve())
                        if key in mapping:raise ValueError('Duplicate path in labels CSV: '+entry['path'])
                        if entry['true_label'] not in labels:raise ValueError('Unknown class: '+entry['true_label'])
                        mapping[key]=entry['true_label']
                matched=0
                for r in rows:
                    key=str(Path(r['path']).resolve())
                    if key in mapping:r['true_label']=mapping[key];matched+=1
                refresh()
                messagebox.showinfo('Labels',f'Matched {matched} of {len(rows)} images.',parent=win)
            except Exception as exc:messagebox.showerror('Labels CSV',str(exc),parent=win)
        ttk.Button(controls,text='Import labels CSV',command=import_csv).pack(side=tk.LEFT)
        def show():
            if any(r['true_label'] not in labels for r in rows):
                messagebox.showwarning('Missing labels','Assign a true label to every successful image before generating the matrix.',parent=win);return
            self.cm_plot(rows,labels,title,failed)
        ttk.Button(controls,text='Show matrix',command=show).pack(side=tk.RIGHT)

    def cm_plot(self, rows, labels, title, failed=0):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        matrix,skipped=matrix_counts(rows,labels)
        count=int(matrix.sum())
        if not count:
            messagebox.showinfo('Confusion matrix','No labeled predictions available.');return
        win=tk.Toplevel(self.root);win.title(title+' — confusion matrix');win.geometry('950x800')
        accuracy=float(np.trace(matrix)/count)
        ttk.Label(win,text=f'{title} | Evaluated: {count} | Excluded: {failed+skipped} | Accuracy on these images: {accuracy:.2%}').pack()
        ttk.Label(win,text='Rows = true labels; columns = predicted labels. Small samples do not estimate overall model accuracy.').pack()
        fig=Figure(figsize=(9,6),tight_layout=True);ax=fig.add_subplot(111)
        im=ax.imshow(matrix,cmap='Blues');fig.colorbar(im,ax=ax)
        ax.set_xticks(range(len(labels)),labels,rotation=45,ha='right');ax.set_yticks(range(len(labels)),labels)
        ax.set_xlabel('Predicted label');ax.set_ylabel('True label')
        for i in range(len(labels)):
            for j in range(len(labels)):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
        canvas=FigureCanvasTkAgg(fig,master=win);canvas.draw();canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        NavigationToolbar2Tk(canvas,win)
        def export():
            path=filedialog.asksaveasfilename(parent=win,defaultextension='.csv',filetypes=[('CSV','*.csv')])
            if path:
                try:
                    with open(path,'w',newline='',encoding='utf-8') as f:
                        writer=csv.writer(f);writer.writerow(['True / Predicted']+labels)
                        writer.writerows([label]+matrix[i].tolist() for i,label in enumerate(labels))
                except OSError as exc:messagebox.showerror('Export',str(exc),parent=win)
        ttk.Button(win,text='Export matrix CSV (toolbar saves PNG)',command=export).pack()

    def cm_dataset(self):
        if self.predictor is None:
            messagebox.showinfo('Model','Wait for the model to load.');return
        folder=filedialog.askdirectory(title='Select dataset root containing class-named folders',initialdir=self.get_config_value('paths','data_dir') or '.')
        if not folder:return
        labels=self.get_config_value('dataset','class_names')
        extensions={s.lower() for s in self.get_config_value('gui','supported_formats')}
        files=[p for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.suffix.lower() in extensions and folder_label(p,labels)]
        if not files:
            messagebox.showinfo('Dataset','No labeled images found. Use folders named exactly like the configured classes.');return
        win=tk.Toplevel(self.root);win.title('Overall dataset evaluation')
        status=tk.StringVar(value=f'Preparing {len(files)} labeled images...')
        ttk.Label(win,textvariable=status,padding=20).pack()
        cancel=threading.Event()
        ttk.Button(win,text='Cancel',command=cancel.set).pack()
        win.protocol('WM_DELETE_WINDOW',cancel.set)
        # Queue keeps every Tk operation on the GUI thread.
        import queue
        events=queue.Queue()
        def work():
            results=[]
            for i,p in enumerate(files,1):
                if cancel.is_set():events.put(('cancel',None));return
                try:
                    r=self.predictor.predict_single(str(p))
                    results.append({'path':str(p),'true_label':folder_label(p,labels),'predicted_class':r['predicted_class']})
                except Exception as exc:results.append({'path':str(p),'error':str(exc)})
                events.put(('progress',i))
            events.put(('done',results))
        def poll():
            try:
                while True:
                    kind,value=events.get_nowait()
                    if kind=='progress':status.set(f'Evaluated {value}/{len(files)}')
                    elif kind=='cancel':win.destroy();return
                    else:
                        win.destroy()
                        self.cm_plot(value,labels,'Overall selected dataset')
                        return
            except queue.Empty:pass
            self.root.after(100,poll)
        threading.Thread(target=work,daemon=True).start();poll()
