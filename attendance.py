# attendance_fixed_layout.py (updated - camera autodetect fixed)
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter, Image
import os
import pyttsx3
import csv
import datetime
import shutil
import time
import traceback
import cv2   # <-- IMPORTANT: cv2 must be imported

# project modules (unchanged)
import show_attendance
import takeImage
import trainImage
import automaticAttedance

# ------------------ Paths ------------------
haarcasecade_path = "haarcascade_frontalface_default.xml"
TRAINING_LABEL_DIR = "TrainingImageLabel"
os.makedirs(TRAINING_LABEL_DIR, exist_ok=True)
trainimagelabel_path = os.path.join(TRAINING_LABEL_DIR, "Trainner.yml")

trainimage_path = "TrainingImage"
os.makedirs(trainimage_path, exist_ok=True)

studentdetail_path = "./StudentDetails/studentdetails.csv"
attendance_path = "Attendance"
os.makedirs(os.path.dirname(studentdetail_path) or ".", exist_ok=True)

# ------------------ Helpers ------------------
def text_to_speech(user_text):
    try:
        engine = pyttsx3.init()
        engine.say(user_text)
        engine.runAndWait()
    except Exception:
        pass

def testVal(inStr, acttyp):
    """
    Validation for Enrollment entry used by tkinter validatecommand.
    Accepts letters, digits, underscore and hyphen (common enrollment formats).
    Does NOT accept spaces. Returns True if value is allowed.
    """
    if acttyp == "1":  # insert action
        if inStr == "":
            return True
        # allow letters, digits, underscore and dash
        import re
        if re.fullmatch(r"[A-Za-z0-9_\-]+", inStr):
            return True
        return False
    return True

def rounded_rect_image(size, radius, fill, shadow=False):
    w, h = size
    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle((0, 0, w, h), radius=radius, fill=fill)
    if shadow:
        shadow_img = Image.new("RGBA", (w + 20, h + 20), (0, 0, 0, 0))
        shdraw = ImageDraw.Draw(shadow_img)
        shdraw.rounded_rectangle((10, 10, w + 10, h + 10), radius=radius + 4, fill=(0, 0, 0, 100))
        shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(8))
        out = Image.new("RGBA", shadow_img.size, (0, 0, 0, 0))
        out.paste(shadow_img, (0, 0))
        out.paste(base, (10, 10), base)
        return out
    return base

def _ensure_student_csv():
    """Ensure the studentdetails CSV exists and has headers."""
    if not os.path.exists(studentdetail_path):
        os.makedirs(os.path.dirname(studentdetail_path) or ".", exist_ok=True)
        with open(studentdetail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Enrollment", "Name", "RegisteredOn"])

def _first_image_timestamp(enroll_id):
    """
    Try to find any training image file that contains the enroll_id in its filename
    and return a human-friendly timestamp string based on file mtime.
    Returns None if no file found.
    """
    try:
        enroll_low = str(enroll_id).lower()
        for root, _, files in os.walk(trainimage_path):
            for fname in files:
                if enroll_low in fname.lower():
                    full = os.path.join(root, fname)
                    try:
                        mtime = os.path.getmtime(full)
                        return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        continue
    except Exception:
        pass
    return None

def _read_all_students():
    """
    Return list of dict rows from CSV (safe).
    If RegisteredOn is missing/empty, try to produce a fallback timestamp using training image file mtime.
    This does NOT modify the CSV file — it only supplies a sensible value for UI display.
    """
    _ensure_student_csv()
    rows = []
    try:
        with open(studentdetail_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                enroll = str(r.get("Enrollment", "") or "").strip()
                name = str(r.get("Name", "") or "").strip()
                reg = str(r.get("RegisteredOn", "") or "").strip()
                if not reg:
                    inferred = _first_image_timestamp(enroll) if enroll else None
                    if inferred:
                        reg = inferred
                    else:
                        reg = "Unknown"
                rows.append({"Enrollment": enroll, "Name": name, "RegisteredOn": reg})
    except Exception:
        return []
    return rows

def _is_enrollment_registered(enroll_id):
    """Return True if enrollment exists in studentdetails CSV."""
    rows = _read_all_students()
    for row in rows:
        if row.get("Enrollment", "").strip().lower() == enroll_id.strip().lower():
            return True
    return False

def _append_student_record(enroll_id, name):
    """Append a new student record to CSV with timestamp."""
    _ensure_student_csv()
    with open(studentdetail_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([enroll_id, name, datetime.datetime.now().isoformat(sep=" ", timespec="seconds")])

# ------------------ Camera autodetect helper ------------------
def open_first_working_camera(max_index=4, timeout_s=5):
    """
    Try to open camera indices 0..max_index using several backends.
    Returns a working cv2.VideoCapture object (already opened) or None.
    Caller must release the returned capture.
    """
    backends = []
    # prefer platform-specific backends if available
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(("CAP_DSHOW", cv2.CAP_DSHOW))
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(("CAP_MSMF", cv2.CAP_MSMF))
    if hasattr(cv2, "CAP_V4L2"):
        backends.append(("CAP_V4L2", cv2.CAP_V4L2))
    # default last
    backends.append(("DEFAULT", None))

    for idx in range(0, max_index + 1):
        for name, backend in backends:
            try:
                if backend is None:
                    cap = cv2.VideoCapture(idx)
                else:
                    cap = cv2.VideoCapture(idx, backend)
                if not cap or not cap.isOpened():
                    try:
                        cap.release()
                    except Exception:
                        pass
                    continue
                # try to read a frame within timeout
                t0 = time.time()
                worked = False
                while time.time() - t0 < timeout_s:
                    ret, _ = cap.read()
                    if ret:
                        worked = True
                        break
                    time.sleep(0.05)
                if worked:
                    # caller will release
                    return cap
                else:
                    try:
                        cap.release()
                    except Exception:
                        pass
            except Exception:
                try:
                    cap.release()
                except Exception:
                    pass
                continue
    return None

# ------------------ Rounded Button ------------------
class RoundedButton(tk.Canvas):
    def __init__(self, master, text="", command=None, width=200, height=40, bg_color=None, corner=12, **kwargs):
        super().__init__(master, width=width, height=height, highlightthickness=0, bd=0)
        self.width = width
        self.height = height
        self.command = command
        self.corner = corner
        self.bg_color = bg_color
        self.default_bg = "#7ad3ff"
        try:
            app = master
            while app and not hasattr(app, "accent"):
                app = getattr(app, "master", None)
            if app and hasattr(app, "accent"):
                self.default_bg = app.accent
        except Exception:
            pass
        self.bg = bg_color if bg_color else "#16202b"
        self.fg = "#052026" if bg_color else "#e6eef8"
        self.text = text
        self._draw_button()
        self.bind("<ButtonRelease-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _draw_button(self):
        self.delete("all")
        rounded = rounded_rect_image((self.width, self.height), self.corner, fill=self._rgb(self.bg), shadow=False)
        self._img = ImageTk.PhotoImage(rounded)
        self.create_image(0, 0, image=self._img, anchor="nw")
        self.create_text(self.width // 2, self.height // 2, text=self.text, font=("Segoe UI Semibold", 11), fill=self.fg)

    def _on_click(self, event):
        if callable(self.command):
            try:
                if self.command.__code__.co_argcount == 1:
                    self.command(text_to_speech)
                else:
                    self.command()
            except Exception:
                try:
                    self.command()
                except Exception:
                    pass

    def _on_enter(self, event):
        self.bg = self._lighten(self.bg, 0.08)
        self.fg = "#052026" if self.bg_color else "#ffffff"
        self._draw_button()

    def _on_leave(self, event):
        self.bg = self.bg_color if self.bg_color else "#16202b"
        self.fg = "#052026" if self.bg_color else "#e6eef8"
        self._draw_button()

    def _rgb(self, hex_color):
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b, 255)
        return (20, 32, 40, 255)

    def _lighten(self, hex_color, amount=0.1):
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = min(255, int(r + (255 - r) * amount))
        g = min(255, int(g + (255 - g) * amount))
        b = min(255, int(b + (255 - b) * amount))
        return f"#{r:02x}{g:02x}{b:02x}"

# ------------------ Main App ------------------
class ModernFaceRecognizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognizer — Modern Dashboard")
        self.geometry("1280x720")
        self.minsize(1000, 640)
        self.configure(bg="#0f1226")
        self.protocol("WM_DELETE_WINDOW", self.on_quit)

        # theme / colors
        self.accent = "#7ad3ff"
        self.card_bg = "#0f1724"
        self.card_faint = "#0b0e17"
        self.text_primary = "#e6eef8"
        self.muted = "#a8b3c7"

        self._build_layout()

    # ---------- helpers ----------
    def _backup_file_timestamped(self, path):
        try:
            stamp = str(int(time.time()))
            bak = f"{path}.bak.{stamp}"
            shutil.copyfile(path, bak)
            return bak
        except Exception:
            return None

    def _remove_student_from_csv(self, enroll_id):
        """Remove rows with Enrollment == enroll_id from CSV; create backup first."""
        _ensure_student_csv()
        backup = self._backup_file_timestamped(studentdetail_path)
        kept_rows = []
        headers = ["Enrollment", "Name", "RegisteredOn"]
        try:
            with open(studentdetail_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    headers = reader.fieldnames
                for row in reader:
                    if row.get("Enrollment", "").strip().lower() != enroll_id.strip().lower():
                        kept_rows.append(row)
        except FileNotFoundError:
            return False, "studentdetails.csv not found."

        try:
            with open(studentdetail_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for r in kept_rows:
                    writer.writerow(r)
            return True, None
        except Exception as e:
            return False, str(e)

    def _delete_training_images_for(self, enroll_id):
        removed = 0
        if not os.path.exists(trainimage_path):
            return removed
        for root, _, files in os.walk(trainimage_path):
            for fname in files:
                try:
                    if enroll_id.lower() in fname.lower():
                        os.remove(os.path.join(root, fname))
                        removed += 1
                except Exception:
                    pass
        return removed

    # ---------- Registered window (legacy) ----------
    def open_registered_students_ui(self):
        rows = _read_all_students()
        popup = tk.Toplevel(self)
        popup.title("Registered Students")
        popup.geometry("760x480")
        popup.configure(bg="#0b0e17")

        header_frame = tk.Frame(popup, bg="#0b0e17", padx=8, pady=8)
        header_frame.pack(fill="x")
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=3)
        header_frame.grid_columnconfigure(2, weight=2)
        header_frame.grid_columnconfigure(3, weight=1)

        tk.Label(header_frame, text="Enrollment", bg="#0b0e17", fg=self.text_primary, font=("Segoe UI Semibold", 11)).grid(row=0, column=0, sticky="w", padx=(6,6))
        tk.Label(header_frame, text="Name", bg="#0b0e17", fg=self.text_primary, font=("Segoe UI Semibold", 11)).grid(row=0, column=1, sticky="w", padx=(6,6))
        tk.Label(header_frame, text="RegisteredOn", bg="#0b0e17", fg=self.text_primary, font=("Segoe UI Semibold", 11)).grid(row=0, column=2, sticky="w", padx=(6,6))
        tk.Label(header_frame, text="Action", bg="#0b0e17", fg=self.text_primary, font=("Segoe UI Semibold", 11)).grid(row=0, column=3, sticky="e", padx=(6,6))

        body_frame = tk.Frame(popup, bg="#0b0e17")
        body_frame.pack(fill="both", expand=True, padx=6, pady=(0,8))
        canvas = tk.Canvas(body_frame, bg="#0b0e17", highlightthickness=0)
        vsb = tk.Scrollbar(body_frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg="#0b0e17")
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        def on_config(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", on_config)

        def make_row(r):
            enroll = (r.get("Enrollment") or "").strip()
            name = (r.get("Name") or "").strip()
            reg_on = (r.get("RegisteredOn") or "").strip()

            row = tk.Frame(inner, bg="#0b0e17", pady=6)
            row.pack(fill="x", padx=4)

            row.grid_columnconfigure(0, weight=1)
            row.grid_columnconfigure(1, weight=3)
            row.grid_columnconfigure(2, weight=2)
            row.grid_columnconfigure(3, weight=1)

            lbl_e = tk.Label(row, text=enroll, bg="#0b0e17", fg=self.text_primary, anchor="w")
            lbl_e.grid(row=0, column=0, sticky="w", padx=(6,6))

            lbl_n = tk.Label(row, text=name, bg="#0b0e17", fg=self.text_primary, anchor="w")
            lbl_n.grid(row=0, column=1, sticky="w", padx=(6,6))

            lbl_t = tk.Label(row, text=reg_on, bg="#0b0e17", fg=self.muted, anchor="w")
            lbl_t.grid(row=0, column=2, sticky="w", padx=(6,6))

            def on_delete():
                if not enroll:
                    messagebox.showinfo("Info", "Invalid enrollment id.")
                    return
                ask = messagebox.askyesno("Confirm", f"Delete record for {enroll}? This removes CSV row and training images.")
                if not ask:
                    return
                ok_csv, err = self._remove_student_from_csv(enroll)
                if not ok_csv:
                    messagebox.showerror("Error", f"Failed to update CSV: {err}")
                    return
                removed = self._delete_training_images_for(enroll)
                # automatic model backup + delete if model exists
                model_removed = None
                if os.path.exists(trainimagelabel_path):
                    bak = self._backup_file_timestamped(trainimagelabel_path)
                    try:
                        os.remove(trainimagelabel_path)
                        model_removed = True
                    except Exception:
                        model_removed = False

                msg = f"Deleted {enroll}."
                if removed:
                    msg += f" Removed {removed} training image(s)."
                if model_removed is True:
                    msg += " Trained model removed (backup created)."
                elif model_removed is False:
                    msg += " Failed to remove trained model."
                messagebox.showinfo("Deleted", msg)
                row.destroy()

            del_btn = tk.Button(row, text="Delete", command=on_delete, bg="#7ad3ff", fg="#04242a", bd=0, padx=8)
            del_btn.grid(row=0, column=3, sticky="e", padx=(6,12))

        if not rows:
            tk.Label(inner, text="No registered students found.", bg="#0b0e17", fg=self.muted).pack(pady=12)
        else:
            for r in rows:
                make_row(r)

        popup.transient(self)
        popup.grab_set()

    def open_manage_students_ui(self):
        """
        Open a scrollable window that lists all registered students with a Delete button at end of each row.
        This replaces the older 'Registered' view and provides per-row deletion + model backup+delete.
        """
        _ensure_student_csv()
        students = []
        try:
            with open(studentdetail_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    students.append({
                        "Enrollment": str(row.get("Enrollment", "")).strip(),
                        "Name": str(row.get("Name", "")).strip(),
                        "RegisteredOn": str(row.get("RegisteredOn", "")).strip()
                    })
        except Exception:
            students = []

        popup = tk.Toplevel(self)
        popup.title("Manage Registered Students")
        popup.geometry("820x460")
        popup.configure(bg="#0b0e17")
        popup.transient(self)
        popup.grab_set()

        header = tk.Frame(popup, bg="#0b0e17", pady=8)
        header.pack(fill="x")
        tk.Label(header, text="Registered students", fg=self.text_primary, bg="#0b0e17",
                 font=("Segoe UI Semibold", 14)).pack(side="left", padx=12)

        info_label = tk.Label(header, text=f"{len(students)} students", fg=self.muted, bg="#0b0e17",
                              font=("Segoe UI", 10))
        info_label.pack(side="right", padx=12)

        container = tk.Frame(popup, bg="#0b0e17")
        container.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        canvas = tk.Canvas(container, bg="#0b0e17", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg="#0b0e17")

        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        hdr = tk.Frame(scrollable, bg="#0b0e17")
        hdr.pack(fill="x", pady=(4, 6))
        tk.Label(hdr, text="Enrollment", width=18, anchor="w", fg=self.text_primary, bg="#0b0e17",
                 font=("Segoe UI Semibold", 10)).grid(row=0, column=0, padx=(6, 6))
        tk.Label(hdr, text="Name", width=30, anchor="w", fg=self.text_primary, bg="#0b0e17",
                 font=("Segoe UI Semibold", 10)).grid(row=0, column=1, padx=(6, 6))
        tk.Label(hdr, text="RegisteredOn", width=28, anchor="w", fg=self.text_primary, bg="#0b0e17",
                 font=("Segoe UI Semibold", 10)).grid(row=0, column=2, padx=(6, 6))
        tk.Label(hdr, text="Action", width=12, anchor="center", fg=self.text_primary, bg="#0b0e17",
                 font=("Segoe UI Semibold", 10)).grid(row=0, column=3, padx=(6, 6))

        row_frames = []

        def make_delete_callback(enroll, row_frame, count_label):
            def _delete():
                confirm = messagebox.askyesno("Confirm delete",
                                              f"Delete student {enroll}? This will remove CSV entry, training images and delete the trained model (backup will be created).")
                if not confirm:
                    return

                ok, err = self._remove_student_from_csv(enroll)
                if not ok:
                    messagebox.showerror("Delete failed", f"Failed to remove CSV row: {err}")
                    return

                removed = self._delete_training_images_for(enroll)

                # automatic timestamped backup + delete of model file if exists
                model_removed = None
                if os.path.exists(trainimagelabel_path):
                    bak = self._backup_file_timestamped(trainimagelabel_path)
                    try:
                        os.remove(trainimagelabel_path)
                        model_removed = True
                    except Exception:
                        model_removed = False

                try:
                    row_frame.destroy()
                except Exception:
                    pass

                try:
                    remaining = sum(1 for rf in row_frames if rf.winfo_exists())
                    count_label.config(text=f"{remaining} students")
                except Exception:
                    pass

                msg = f"Deleted {enroll} from CSV."
                if removed:
                    msg += f" Removed {removed} training image(s)."
                if model_removed is True:
                    msg += " Trained model removed (backup created)."
                elif model_removed is False:
                    msg += " Failed to remove trained model."
                messagebox.showinfo("Deleted", msg)
            return _delete

        for idx, s in enumerate(students):
            rf = tk.Frame(scrollable, bg="#0b0e17", pady=6)
            rf.pack(fill="x", padx=2)
            row_frames.append(rf)

            e_lbl = tk.Label(rf, text=s["Enrollment"], width=18, anchor="w", fg="#e6eef8", bg="#0b0e17",
                             font=("Segoe UI", 10))
            e_lbl.grid(row=0, column=0, padx=(6, 6))
            n_lbl = tk.Label(rf, text=s["Name"], width=30, anchor="w", fg="#e6eef8", bg="#0b0e17",
                             font=("Segoe UI", 10))
            n_lbl.grid(row=0, column=1, padx=(6, 6))
            r_lbl = tk.Label(rf, text=s["RegisteredOn"], width=28, anchor="w", fg="#e6eef8", bg="#0b0e17",
                             font=("Segoe UI", 10))
            r_lbl.grid(row=0, column=2, padx=(6, 6))

            del_btn = tk.Button(rf, text="Delete", width=10, bg="#7ad3ff", fg="#04242a",
                                command=make_delete_callback(s["Enrollment"], rf, info_label), bd=0,
                                font=("Segoe UI Semibold", 9))
            del_btn.grid(row=0, column=3, padx=(12, 8))

        if not students:
            empty = tk.Label(scrollable, text="No registered students found.", fg="#a8b3c7", bg="#0b0e17",
                             font=("Segoe UI", 11), pady=12)
            empty.pack()

        footer = tk.Frame(popup, bg="#0b0e17")
        footer.pack(fill="x", pady=(0, 12))
        close_btn = tk.Button(footer, text="Close", command=popup.destroy, bg="#111418", fg="yellow",
                              font=("Segoe UI", 11), bd=0, padx=12, pady=6)
        close_btn.pack(side="right", padx=12)

    def on_quit(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to close?"):
            self.destroy()

    def _build_layout(self):
        # top header
        header = tk.Frame(self, bg=self["bg"], padx=20, pady=10)
        header.pack(fill="x")

        logo = None
        try:
            logo_img = Image.open("UI_Image/nitk_logo.png").resize((56,56), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo = tk.Label(header, image=self.logo_photo, bg=self["bg"])
        except Exception:
            logo = tk.Label(header, text="NITK", fg=self.accent, bg=self["bg"], font=("Segoe UI Semibold", 18))
        logo.grid(row=0, column=0, rowspan=2, sticky="w", padx=(4,12))

        title = tk.Label(header, text="Department of IT", fg=self.text_primary, bg=self["bg"], font=("Segoe UI Semibold", 26))
        title.grid(row=0, column=1, sticky="w")
        subtitle = tk.Label(header, text="Welcome to NIT Karnataka, Surathkal", fg=self.muted, bg=self["bg"], font=("Segoe UI", 11))
        subtitle.grid(row=1, column=1, sticky="w", pady=(2,0))

        # Main area
        content = tk.Frame(self, bg=self["bg"], padx=20, pady=12)
        content.pack(fill="both", expand=True)

        content.grid_columnconfigure(0, weight=2)
        content.grid_columnconfigure(1, weight=1)

        # LEFT
        left_card = tk.Frame(content, bg=self.card_bg)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0,12), pady=6)
        left_card.grid_rowconfigure(1, weight=1)
        left_inner = tk.Frame(left_card, bg=self.card_bg, padx=22, pady=18)
        left_inner.pack(fill="both", expand=True)

        hero_title = tk.Label(left_inner, text="Attendance Dashboard", fg=self.text_primary, bg=self.card_bg, font=("Segoe UI Semibold", 20))
        hero_title.pack(anchor="nw")
        hero_desc = tk.Label(left_inner, text="Fast. Accurate. Minimal. Register faces, train the model and start attendance in seconds.",
                             fg=self.muted, bg=self.card_bg, wraplength=760, justify="left", font=("Segoe UI", 10))
        hero_desc.pack(anchor="nw", pady=(6,16))

        tiles_frame = tk.Frame(left_inner, bg=self.card_bg)
        tiles_frame.pack(fill="x", pady=(6,12))
        tiles_frame.grid_columnconfigure((0,1,2), weight=1, uniform="tile")

        tile_def = [
            ("UI_Image/register.png", "Register a new student", self.open_take_image_ui),
            ("UI_Image/verifyy.png", "Take Attendance", lambda: automaticAttedance.subjectChoose(text_to_speech)),
            ("UI_Image/attendance.png", "View Attendance", lambda: show_attendance.subjectchoose(text_to_speech)),
        ]

        self._tile_images = []
        for idx, (path, label_text, cmd) in enumerate(tile_def):
            card = tk.Frame(tiles_frame, bg=self.card_faint, padx=12, pady=10)
            card.grid(row=0, column=idx, padx=6, sticky="nsew")
            try:
                icon = Image.open(path).resize((86,62), Image.LANCZOS)
                icon_photo = ImageTk.PhotoImage(icon)
                self._tile_images.append(icon_photo)
                icon_lbl = tk.Label(card, image=icon_photo, bg=self.card_faint)
                icon_lbl.pack(anchor="w")
            except Exception:
                icon_lbl = tk.Label(card, text="Icon", bg=self.card_faint, fg=self.text_primary)
                icon_lbl.pack(anchor="w")
            lbl = tk.Label(card, text=label_text, bg=self.card_faint, fg=self.text_primary, font=("Segoe UI Semibold", 11))
            lbl.pack(anchor="w", pady=(8,6))
            btn = RoundedButton(card, text="Open", command=cmd, width=100, height=36, bg_color=self.accent)
            btn.pack(anchor="w")

        preview = tk.Frame(left_inner, bg="#0b0e17", height=300)
        preview.pack(fill="both", expand=True, pady=(8,0))
        preview.pack_propagate(False)
        preview_label = tk.Label(preview, text="Preview / Logs", bg="#0b0e17", fg=self.muted)
        preview_label.pack(anchor="nw", padx=8, pady=8)

        # RIGHT
        right_card = tk.Frame(content, bg=self.card_bg)
        right_card.grid(row=0, column=1, sticky="nsew", pady=6)
        right_inner = tk.Frame(right_card, bg=self.card_bg, padx=18, pady=18)
        right_inner.pack(fill="both", expand=True)

        control_buttons = [
            ("Manage Students", self.open_manage_students_ui),
            ("Register a new student", self.open_take_image_ui),
            ("Train Images (Model)", lambda: trainImage.TrainImage(haarcasecade_path, trainimage_path, trainimagelabel_path, None, text_to_speech)),
            ("Take Attendance", lambda: automaticAttedance.subjectChoose(text_to_speech)),
            ("View Attendance", lambda: show_attendance.subjectchoose(text_to_speech)),
            ("Exit", self.on_quit)
        ]

        for i, (t, cmd) in enumerate(control_buttons):
            rb = RoundedButton(right_inner, text=t, command=cmd, width=260, height=46, bg_color=None)
            rb.pack(pady=(6 if i == 0 else 12, 0))

        footer = tk.Label(right_inner, text="Built for quick lab use • Keep cascade and UI_Image folder intact",
                          fg=self.muted, bg=self.card_bg, wraplength=260, justify="left")
        footer.pack(side="bottom", pady=12)

    # popup to register/take image
    def open_take_image_ui(self):
        ImageUI = tk.Toplevel(self)
        ImageUI.title("Take Student Image")
        ImageUI.geometry("780x480")
        ImageUI.configure(bg="#0b0e17")
        ImageUI.resizable(0, 0)

        header = tk.Label(ImageUI, text="Register Your Face", font=("Segoe UI Semibold", 22), fg=self.text_primary, bg="#0b0e17")
        header.pack(pady=(16, 6))

        form_frame = tk.Frame(ImageUI, bg="#0b0e17")
        form_frame.pack(pady=10, padx=12)

        lbl1 = tk.Label(form_frame, text="Enrollment No", fg=self.text_primary, bg="#0b0e17", font=("Segoe UI", 11))
        lbl1.grid(row=0, column=0, sticky="e", padx=(6, 12), pady=10)
        txt1 = tk.Entry(form_frame, width=20, font=("Segoe UI", 14), bg="#111418", fg=self.text_primary, bd=1, relief="solid", insertbackground=self.text_primary)
        txt1.grid(row=0, column=1, padx=(2, 6), pady=10)
        reg = ImageUI.register(testVal)
        txt1.config(validate="key", validatecommand=(reg, "%P", "%d"))

        lbl2 = tk.Label(form_frame, text="Name", fg=self.text_primary, bg="#0b0e17", font=("Segoe UI", 11))
        lbl2.grid(row=1, column=0, sticky="e", padx=(6, 12), pady=10)
        txt2_var = tk.StringVar()
        txt2 = tk.Entry(form_frame, width=20, font=("Segoe UI", 14), textvariable=txt2_var, bg="#111418", fg=self.text_primary, bd=1, relief="solid", insertbackground=self.text_primary)
        txt2.grid(row=1, column=1, padx=(2, 6), pady=10)

        lbl3 = tk.Label(form_frame, text="Notification", fg=self.text_primary, bg="#0b0e17", font=("Segoe UI", 11))
        lbl3.grid(row=2, column=0, sticky="ne", padx=(6, 12), pady=10)
        message = tk.Label(form_frame, text="", width=34, fg=self.muted, bg="#0b0e17", font=("Segoe UI", 10), anchor="w")
        message.grid(row=2, column=1, padx=(2, 6), pady=10)

        def take_image():
            l1 = txt1.get().strip()
            l2 = txt2_var.get().strip()

            # final server-side validation (defensive)
            if l1 == "" or l2 == "":
                message.config(text="Enrollment & Name required!!!", fg="red")
                messagebox.showwarning("Warning", "Enrollment & Name required!!!")
                return

            # check characters again
            import re
            if not re.fullmatch(r"[A-Za-z0-9_\-]+", l1):
                message.config(text="Enrollment contains invalid characters. Use letters, digits, _ or -", fg="red")
                return

            # check camera availability before calling takeImage()
            cap = None
            try:
                cap = open_first_working_camera(max_index=3, timeout_s=2)
                if cap is None:
                    message.config(text="Camera error: Unable to open camera. Check camera permissions/other apps.", fg="red")
                    messagebox.showerror("Camera Error", "Unable to open camera. Close other apps using the webcam and ensure OS privacy settings allow access.")
                    return
                # we found a working camera; release it immediately because takeImage likely opens its own capture.
                try:
                    cap.release()
                except Exception:
                    pass
            except Exception as e:
                message.config(text=f"Camera error: {e}", fg="red")
                messagebox.showerror("Camera Error", f"Camera error: {e}")
                return

            # ensure folder for model exists
            try:
                os.makedirs(os.path.dirname(trainimagelabel_path) or ".", exist_ok=True)
            except Exception:
                pass

            success = False
            try:
                # call the original capture routine (this opens the camera internally)
                takeImage.TakeImage(l1, l2, haarcasecade_path, trainimage_path, message,
                                    lambda: messagebox.showwarning("Warning", "Enrollment & Name required!!!"),
                                    text_to_speech)
                # If TakeImage succeeded, append to CSV if not present
                try:
                    if not _is_enrollment_registered(l1):
                        _append_student_record(l1, l2)
                        message.config(text=f"Enrollment {l1} registered.", fg="green")
                        text_to_speech(f"Enrollment number {l1} registered successfully")
                    else:
                        message.config(text=f"Enrollment {l1} already present.", fg="orange")
                except Exception as e:
                    message.config(text=f"Registered, but failed to update CSV: {e}", fg="orange")
                success = True
            except Exception as e:
                tb = traceback.format_exc()
                message.config(text=f"Capture failed: {e}", fg="red")
                messagebox.showerror("Capture failed", f"Capture failed: {e}\n\nSee console for traceback.")
                print("TakeImage error:", tb)
            finally:
                # clear inputs only if success
                if success:
                    try:
                        txt1.delete(0, "end")
                        txt2_var.set("")
                    except Exception:
                        pass

        btns = tk.Frame(ImageUI, bg="#0b0e17")
        btns.pack(pady=(12, 8))
        takeBtn = RoundedButton(btns, text="Take Image", command=take_image, width=160, height=44, bg_color=self.accent)
        takeBtn.pack(side="left", padx=8)
        trainBtn = RoundedButton(btns, text="Train Image", command=lambda: trainImage.TrainImage(haarcasecade_path, trainimage_path, trainimagelabel_path, message, text_to_speech), width=160, height=44, bg_color=None)
        trainBtn.pack(side="left", padx=8)

        def close_popup():
            ImageUI.destroy()

        exit_btn = RoundedButton(btns, text="Exit", command=close_popup, width=90, height=44, bg_color=None)
        exit_btn.pack(side="left", padx=8)

# ------------------ Run ------------------
if __name__ == "__main__":
    app = ModernFaceRecognizerApp()
    app.mainloop()
