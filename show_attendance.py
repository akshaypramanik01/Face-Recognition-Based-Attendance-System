import os
import csv
import tkinter as tk
from tkinter import ttk, messagebox
from glob import glob
import pandas as pd
from functools import reduce

def subjectchoose(text_to_speech):
    # Colors consistent with new GUI
    BG = "#0f1724"
    CARD = "#0b0e17"
    TEXT = "#e6eef8"
    MUTED = "#a8b3c7"
    ACCENT = "#7ad3ff"

    def calculate_attendance(subject, status_label):
        if not subject:
            status_label.config(text="Please enter the subject name.", fg="orange")
            text_to_speech("Please enter the subject name")
            return

        folder = os.path.join("Attendance", subject)
        pattern = os.path.join(folder, f"{subject}*.csv")
        filenames = glob(pattern)

        if not filenames:
            status_label.config(text=f"No attendance files found for '{subject}'.", fg="orange")
            text_to_speech(f"No attendance records for {subject}")
            return

        try:
            dfs = [pd.read_csv(f) for f in filenames if os.path.getsize(f) > 0]
            if not dfs:
                status_label.config(text="Found files but none contain data.", fg="orange")
                return

            # Normalize columns: ensure Enrollment and Name exist and are strings
            for df in dfs:
                # Coerce columns to strings to avoid type mismatch issues
                df.columns = [str(c) for c in df.columns]
                if "Enrollment" not in df.columns:
                    raise RuntimeError("CSV missing 'Enrollment' column")
                if "Name" not in df.columns:
                    df["Name"] = ""
                # ensure Enrollment and Name are strings
                df["Enrollment"] = df["Enrollment"].astype(str)
                df["Name"] = df["Name"].astype(str)

            # Merge on Enrollment + Name to align rows across CSVs
            def merge_outer(a, b):
                return pd.merge(a, b, on=["Enrollment", "Name"], how="outer")

            merged = reduce(merge_outer, dfs)
            merged.fillna(0, inplace=True)

            # Identify date columns (all except Enrollment, Name, Attendance)
            cols_to_exclude = {"Enrollment", "Name", "Attendance"}
            date_cols = [c for c in merged.columns if c not in cols_to_exclude]

            # Convert date columns to numeric (0/1) if necessary
            for c in date_cols:
                merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype(int)

            # Compute attendance percentage (mean across date columns)
            if date_cols:
                merged["Attendance"] = (merged[date_cols].mean(axis=1) * 100).round().astype(int).astype(str) + "%"
            else:
                merged["Attendance"] = "0%"

            out_path = os.path.join(folder, "attendance.csv")
            merged.to_csv(out_path, index=False)

            status_label.config(text=f"Attendance calculated — saved to attendance.csv", fg="green")
            text_to_speech(f"Attendance calculated for {subject}")

            # show results in a new window (scrollable Treeview)
            show_attendance_window(subject, merged)

        except Exception as e:
            status_label.config(text=f"Failed to calculate attendance: {e}", fg="red")
            text_to_speech("Failed to calculate attendance")
            print("Attendance calc error:", e)

    def show_attendance_window(subject, df):
        w = tk.Toplevel(root)
        w.title(f"Attendance — {subject}")
        w.configure(bg=BG)
        w.geometry("900x600")

        # Frame for Treeview + scrollbar
        frame = tk.Frame(w, bg=BG, padx=10, pady=10)
        frame.pack(fill="both", expand=True)

        cols = list(df.columns)
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=25)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscroll=vsb.set, xscroll=hsb.set)

        # Setup headings - tighter widths, center headings
        for c in cols:
            tree.heading(c, text=c)
            # make Enrollment and Name a bit wider; date columns smaller
            if c.lower() in ("enrollment",):
                tree.column(c, width=140, anchor="center")
            elif c.lower() in ("name",):
                tree.column(c, width=220, anchor="w")
            elif c.lower() == "attendance":
                tree.column(c, width=90, anchor="center")
            else:
                tree.column(c, width=110, anchor="center")

        # Insert rows: coerce to plain strings (avoid list/NaN display)
        for _, row in df.iterrows():
            values = []
            for c in cols:
                v = row.get(c, "")
                # if it's an ndarray or list (rare), join elements
                if hasattr(v, "tolist") and not isinstance(v, str):
                    try:
                        v = ", ".join(map(str, v.tolist()))
                    except Exception:
                        v = str(v)
                values.append("" if pd.isna(v) else str(v))
            tree.insert("", "end", values=values)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Close button
        close_btn = tk.Button(w, text="Close", command=w.destroy, bg="#111418", fg="yellow", font=("Segoe UI", 11), bd=0, padx=8, pady=6)
        close_btn.pack(pady=8)

        # Make window modal-ish
        w.transient(root)
        w.grab_set()
        w.focus_force()

    # Main popup (modern style)
    root = tk.Toplevel()
    root.title("Subject...")
    root.geometry("620x360")
    root.resizable(False, False)
    root.configure(bg=BG)

    header = tk.Frame(root, bg=BG, pady=10)
    header.pack(fill="x")
    title = tk.Label(header, text="Check Attendance(Individual)", bg=BG, fg=ACCENT, font=("Segoe UI Semibold", 18))
    title.pack(anchor="w", padx=12)

    card = tk.Frame(root, bg=CARD, padx=12, pady=12)
    card.pack(fill="both", expand=True, padx=12, pady=6)

    lbl = tk.Label(card, text="Enter Subject", bg=CARD, fg=TEXT, font=("Segoe UI", 12))
    lbl.grid(row=0, column=0, sticky="w", pady=(6,8), padx=(6,12))

    tx = tk.Entry(card, font=("Segoe UI", 20, "bold"), bg="#111418", fg=TEXT, bd=1, relief="solid", width=18, justify="center", insertbackground=TEXT)
    tx.grid(row=0, column=1, sticky="w", pady=(6,8))

    status_label = tk.Label(card, text="", bg=CARD, fg=MUTED, font=("Segoe UI", 10))
    status_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4,10), padx=6)

    btn_frame = tk.Frame(card, bg=CARD)
    btn_frame.grid(row=2, column=0, columnspan=2, pady=(8,0))

    def on_check_sheets():
        sub = tx.get().strip()
        if not sub:
            status_label.config(text="Please enter the subject name.", fg="orange")
            text_to_speech("Please enter the subject name")
            return
        path = os.path.join("Attendance", sub)
        if not os.path.exists(path):
            status_label.config(text="No sheets for this subject yet", fg="orange")
            return
        try:
            os.startfile(path)
        except Exception:
            status_label.config(text="Failed to open folder", fg="red")

    check_btn = tk.Button(btn_frame, text="Check Sheets", command=on_check_sheets, bg="#111418", fg="yellow", font=("Segoe UI", 11), bd=0, padx=12, pady=8)
    check_btn.grid(row=0, column=0, padx=(0,12))

    fill_btn = tk.Button(btn_frame, text="View Attendance", command=lambda: calculate_attendance(tx.get().strip(), status_label), bg=ACCENT, fg="#04242a", font=("Segoe UI Semibold", 11), bd=0, padx=12, pady=8)
    fill_btn.grid(row=0, column=1, padx=(0,12))

    close_btn = tk.Button(btn_frame, text="Exit", command=root.destroy, bg="#111418", fg="yellow", font=("Segoe UI", 11), bd=0, padx=12, pady=8)
    close_btn.grid(row=0, column=2)

    # Compact footer: show folder path truncated + 'Open' button
    footer_frame = tk.Frame(card, bg=CARD)
    footer_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=(12,0), padx=6)

    attendance_dir = os.path.abspath("Attendance")
    # truncated display to avoid layout break
    short_path = attendance_dir
    if len(short_path) > 70:
        short_path = "..." + short_path[-67:]

    folder_label = tk.Label(footer_frame, text=f"Attendance directory: {short_path}", fg=MUTED, bg=CARD, font=("Segoe UI", 8), anchor="w")
    folder_label.pack(side="left", padx=(0,8))

    def open_attendance_root():
        try:
            os.startfile(attendance_dir)
        except Exception:
            status_label.config(text="Failed to open attendance folder", fg="red")

    open_btn = tk.Button(footer_frame, text="Open Folder", command=open_attendance_root, bg="#111418", fg="yellow", font=("Segoe UI", 9), bd=0, padx=8, pady=4)
    open_btn.pack(side="left")

    tx.focus_set()
    root.transient()  # tie to main window if any
    root.grab_set()
