# attendance_gui.py  — updated full file
import os
import csv
import time
import datetime
import shutil
import ssl
import mimetypes
import smtplib
from email.message import EmailMessage
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import cv2
import pandas as pd
from pathlib import Path


# Paths / constants
haarcasecade_path = "haarcascade_frontalface_default.xml"
TRAINING_LABEL_DIR = "TrainingImageLabel"
trainimagelabel_path = os.path.join(TRAINING_LABEL_DIR, "Trainner.yml")
label_map_path = os.path.join(TRAINING_LABEL_DIR, "label_map.csv")  # optional mapping file
trainimage_path = "TrainingImage"
studentdetail_path = os.path.join("StudentDetails", "studentdetails.csv")
attendance_path = "Attendance"

# Ensure base directories exist
os.makedirs(trainimage_path, exist_ok=True)
os.makedirs(os.path.dirname(studentdetail_path) or ".", exist_ok=True)
os.makedirs(os.path.dirname(trainimagelabel_path) or ".", exist_ok=True)
os.makedirs(attendance_path, exist_ok=True)


# ---------- Helpers for student CSV ----------
def _ensure_student_csv():
    """Ensure studentdetails CSV exists and has headers."""
    if not os.path.exists(studentdetail_path):
        os.makedirs(os.path.dirname(studentdetail_path) or ".", exist_ok=True)
        with open(studentdetail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Enrollment", "Name", "RegisteredOn"])


def _read_students_df():
    """Return a pandas DataFrame of student details (strings)."""
    _ensure_student_csv()
    try:
        df = pd.read_csv(studentdetail_path, dtype=str).fillna("")
        # normalize column names
        expected = ["Enrollment", "Name", "RegisteredOn"]
        for c in expected:
            if c not in df.columns:
                df[c] = ""
        return df[expected].astype(str)
    except Exception:
        # fallback: empty dataframe
        return pd.DataFrame(columns=["Enrollment", "Name", "RegisteredOn"], dtype=str)


def _append_student_record(enroll_id: str, name: str):
    """Append a new student record to CSV with timestamp."""
    _ensure_student_csv()
    registered_on = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
    with open(studentdetail_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([str(enroll_id).strip(), str(name).strip(), registered_on])


def _remove_student_from_csv(enroll_id: str):
    """Remove rows where Enrollment equals enroll_id. Create timestamped backup."""
    _ensure_student_csv()
    bak = None
    try:
        ts = str(int(time.time()))
        bak = studentdetail_path + f".bak.{ts}"
        shutil.copyfile(studentdetail_path, bak)
    except Exception:
        bak = None

    kept = []
    headers = ["Enrollment", "Name", "RegisteredOn"]
    try:
        with open(studentdetail_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                headers = reader.fieldnames
            for row in reader:
                if str(row.get("Enrollment", "")).strip().lower() != str(enroll_id).strip().lower():
                    kept.append(row)
    except FileNotFoundError:
        return False, "studentdetails.csv not found"

    try:
        with open(studentdetail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in kept:
                writer.writerow(r)
        return True, None
    except Exception as e:
        return False, str(e)


def _delete_training_images_for(enroll_id: str):
    """Delete any training image files that contain enroll_id (case-insensitive) in their filename."""
    removed = 0
    if not os.path.exists(trainimage_path):
        return removed
    for root, _, files in os.walk(trainimage_path):
        for fname in files:
            try:
                if str(enroll_id).lower() in fname.lower():
                    os.remove(os.path.join(root, fname))
                    removed += 1
            except Exception:
                pass
    return removed


# ---------- Email helper (simple interactive) ----------
def _interactive_smtp_dialog(parent):
    """
    Obtain sender email, password, SMTP server and port from user via dialog.
    Returns (sender, password, smtp_server, smtp_port) or (None, None, None, None) on cancel.
    """
    dlg = tk.Toplevel(parent)
    dlg.title("SMTP credentials")
    dlg.geometry("480x220")
    dlg.transient(parent)
    dlg.grab_set()

    tk.Label(dlg, text="Sender email:").pack(anchor="w", padx=12, pady=(12, 2))
    sender_var = tk.StringVar(value="pramanikakshay10@gmail.com")
    tk.Entry(dlg, textvariable=sender_var, width=64).pack(padx=12)

    tk.Label(dlg, text="Password (app password recommended):").pack(anchor="w", padx=12, pady=(8, 2))
    pass_var = tk.StringVar(value="segq uzfd pjze qwnk")
    tk.Entry(dlg, textvariable=pass_var, show="*", width=64).pack(padx=12)

    tk.Label(dlg, text="SMTP server (default smtp.gmail.com):").pack(anchor="w", padx=12, pady=(8, 2))
    srv_var = tk.StringVar(value=os.environ.get("SMTP_SERVER", "smtp.gmail.com"))
    tk.Entry(dlg, textvariable=srv_var, width=48).pack(padx=12)

    tk.Label(dlg, text="SMTP port (default 587 = STARTTLS):").pack(anchor="w", padx=12, pady=(8, 2))
    port_var = tk.IntVar(value=int(os.environ.get("SMTP_PORT", 587)))
    tk.Entry(dlg, textvariable=port_var, width=12).pack(padx=12)

    done = {"ok": False}

    def on_ok():
        done["ok"] = True
        dlg.destroy()

    def on_cancel():
        dlg.destroy()

    btnf = tk.Frame(dlg)
    btnf.pack(pady=12)
    tk.Button(btnf, text="OK", width=12, command=on_ok).pack(side="left", padx=8)
    tk.Button(btnf, text="Cancel", width=12, command=on_cancel).pack(side="left", padx=8)

    parent.wait_window(dlg)
    if not done["ok"]:
        return None, None, None, None
    return sender_var.get().strip(), pass_var.get(), srv_var.get().strip(), int(port_var.get())


def _send_email_with_attachment(sender, password, smtp_server, smtp_port, recipient, subject_text, body_text, file_path):
    """Send email with attachment using STARTTLS (port 587) or direct SSL if port==465."""
    try:
        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = subject_text
        msg.set_content(body_text)

        if file_path and os.path.exists(file_path):
            ctype, _ = mimetypes.guess_type(file_path)
            if ctype is None:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)
            with open(file_path, "rb") as f:
                data = f.read()
            msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=os.path.basename(file_path))

        # choose SSL for 465, STARTTLS for 587/other
        if smtp_port == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context, timeout=20) as server:
                server.login(sender, password)
                server.send_message(msg)
        else:
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_server, smtp_port, timeout=20) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(sender, password)
                server.send_message(msg)
        return True, None
    except Exception as exc:
        return False, str(exc)


# ---------- Main subjectChoose UI ----------
def subjectChoose(text_to_speech):
    import subprocess

    def _open_folder(path):
        try:
            if os.name == "nt":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            message_label.config(text=f"Failed to open folder: {e}", fg="red")

    win = tk.Toplevel()
    win.title("Attendance - Subject")
    win.geometry("720x460")
    win.resizable(False, False)
    win.configure(bg="#0f1226")

    # style constants
    CARD_BG = "#0f1724"
    CARD = "#0b0e17"
    TEXT = "#e6eef8"
    MUTED = "#a8b3c7"
    ACCENT = "#7ad3ff"

    header = tk.Frame(win, bg=win["bg"], padx=16, pady=12)
    header.pack(fill="x")
    tk.Label(header, text="Check / Fill Attendance", fg=TEXT, bg=win["bg"], font=("Segoe UI Semibold", 18)).pack(anchor="w")

    card = tk.Frame(win, bg=CARD_BG, padx=18, pady=14)
    card.pack(fill="both", expand=True, padx=16, pady=(6, 16))

    # Subject entry
    tk.Label(card, text="Subject", fg=TEXT, bg=CARD_BG, font=("Segoe UI", 12)).grid(row=0, column=0, sticky="w", padx=(0, 12), pady=(4, 8))
    subject_var = tk.StringVar()
    entry = tk.Entry(card, textvariable=subject_var, font=("Segoe UI", 20, "bold"), bg="#111418", fg=TEXT, bd=1, relief="solid", width=20, justify="center", insertbackground=TEXT)
    entry.grid(row=0, column=1, sticky="w", pady=(4, 8))

    # Faculty email (optional)
    tk.Label(card, text="Faculty email (optional)", fg=TEXT, bg=CARD_BG, font=("Segoe UI", 11)).grid(row=1, column=0, sticky="w", padx=(0, 12), pady=(6, 6))
    faculty_email_var = tk.StringVar()
    faculty_entry = tk.Entry(card, textvariable=faculty_email_var, font=("Segoe UI", 12), bg="#111418", fg=TEXT, bd=1, relief="solid", width=36, insertbackground=TEXT)
    faculty_entry.grid(row=1, column=1, sticky="w", pady=(6, 6))
    faculty_entry.insert(0, "")

    message_label = tk.Label(card, text="", fg=MUTED, bg=CARD_BG, font=("Segoe UI", 10), anchor="w")
    message_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 10))

    btn_frame = tk.Frame(card, bg=CARD_BG)
    btn_frame.grid(row=3, column=0, columnspan=2, pady=(6, 0))

    Path(attendance_path).mkdir(parents=True, exist_ok=True)

    # helper to load student map Enrollment -> Name quickly
    def _get_student_map():
        df = _read_students_df()
        # ensure Enrollment values are str and unique mapping
        mapping = {}
        for _, r in df.iterrows():
            en = str(r["Enrollment"]).strip()
            nm = str(r["Name"]).strip()
            if en:
                mapping[en] = nm
        return mapping

    def calculate_attendance_and_show(subject):
        """Merge all per-session CSVs for subject into attendance.csv and show result in treeview."""
        if not subject:
            message_label.config(text="Please enter subject", fg="orange")
            return None

        folder = os.path.join(attendance_path, subject)
        pattern = os.path.join(folder, f"{subject}*.csv")
        files = [f for f in sorted(Path(folder).glob(f"{subject}*.csv")) if f.is_file()]
        if not files:
            message_label.config(text=f"No attendance files found for {subject}", fg="orange")
            return None

        try:
            dfs = []
            for f in files:
                try:
                    d = pd.read_csv(f, dtype=str).fillna("0")
                    # ensure columns exist
                    if "Enrollment" not in d.columns:
                        raise RuntimeError(f"{f} missing 'Enrollment' column")
                    if "Name" not in d.columns:
                        d["Name"] = ""
                    dfs.append(d)
                except Exception:
                    # skip malformed files but warn
                    print("Skipping malformed attendance file:", f)
            if not dfs:
                message_label.config(text="Found files but none usable", fg="orange")
                return None

            # merge outer on Enrollment + Name
            from functools import reduce
            merged = reduce(lambda a, b: pd.merge(a, b, on=["Enrollment", "Name"], how="outer"), dfs)
            merged = merged.fillna(0)

            # date columns = all columns except Enrollment, Name, Attendance
            excl = {"Enrollment", "Name", "Attendance"}
            date_cols = [c for c in merged.columns if c not in excl]
            for c in date_cols:
                merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype(int)

            if date_cols:
                merged["Attendance"] = (merged[date_cols].mean(axis=1) * 100).round().astype(int).astype(str) + "%"
            else:
                merged["Attendance"] = "0%"

            out_path = os.path.join(folder, "attendance.csv")
            os.makedirs(folder, exist_ok=True)
            merged.to_csv(out_path, index=False)
            message_label.config(text=f"Attendance calculated and saved to {out_path}", fg="green")
            try:
                text_to_speech(f"Attendance calculated for {subject}")
            except Exception:
                pass

            # show results in Treeview window
            _show_attendance_window(subject, merged)
            return out_path
        except Exception as exc:
            message_label.config(text=f"Failed to compute attendance: {exc}", fg="red")
            print("Attendance calc error:", exc)
            return None

    def _show_attendance_window(subject, df):
        w = tk.Toplevel(win)
        w.title(f"Attendance — {subject}")
        w.geometry("900x600")
        w.configure(bg="#0f1724")

        frame = tk.Frame(w, bg="#0f1724", padx=10, pady=10)
        frame.pack(fill="both", expand=True)

        cols = list(df.columns)
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=25)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscroll=vsb.set, xscroll=hsb.set)

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor="center")

        for _, r in df.iterrows():
            vals = [str(r[c]) if not pd.isna(r[c]) else "" for c in cols]
            tree.insert("", "end", values=vals)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Buttons: Email + Close
        btnf = tk.Frame(w, bg="#0f1724")
        btnf.pack(fill="x", pady=8)
        def on_email():
            out_path = os.path.join(attendance_path, subject, "attendance.csv")
            if not os.path.exists(out_path):
                messagebox.showwarning("No file", "Attendance file not found. Please calculate first.")
                return
            recipient = faculty_email_var.get().strip()
            if not recipient:
                recipient = simpledialog.askstring("Recipient", "Enter faculty email:", parent=w)
                if not recipient:
                    return
            creds = _interactive_smtp_dialog(w)
            if not creds or not creds[0]:
                messagebox.showinfo("Canceled", "Email canceled (no SMTP credentials).")
                return
            sender, password, smtp_srv, smtp_port = creds
            ok, err = _send_email_with_attachment(sender, password, smtp_srv, smtp_port, recipient,
                                                  f"Attendance — {subject}", f"Attached attendance for {subject}", out_path)
            if ok:
                messagebox.showinfo("Sent", f"Attendance emailed to {recipient}")
            else:
                messagebox.showerror("Failed", f"Email failed: {err}")

        tk.Button(btnf, text="Email to Faculty", command=on_email, bg="#7ad3ff", fg="#04242a", bd=0, padx=10, pady=6).pack(side="left", padx=8)
        tk.Button(btnf, text="Close", command=w.destroy, bg="#111418", fg="yellow", bd=0, padx=10, pady=6).pack(side="right", padx=8)

        w.transient(win)
        w.grab_set()
        w.focus_force()

    # -------- camera-based FillAttendance implementation ----------
    def FillAttendance_camera():
        Subject = subject_var.get().strip()
        faculty_email = faculty_email_var.get().strip()
        if Subject == "":
            message_label.config(text="Please enter subject name", fg="red")
            try:
                text_to_speech("Please enter the subject name")
            except Exception:
                pass
            return

        # read student data map
        students_df = _read_students_df()
        student_map = {str(r["Enrollment"]).strip(): str(r["Name"]).strip() for _, r in students_df.iterrows()}

        # prepare recognizer
        recognizer = None
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        except Exception:
            message_label.config(text="OpenCV face recognizer not available", fg="red")
            return

        if not os.path.exists(trainimagelabel_path):
            message_label.config(text="Model not found. Please train model first.", fg="yellow")
            try:
                text_to_speech("Model not found. Please train model first.")
            except Exception:
                pass
            return

        try:
            recognizer.read(trainimagelabel_path)
        except Exception as re:
            message_label.config(text=f"Failed loading model: {re}", fg="red")
            return

        if not os.path.exists(haarcasecade_path):
            message_label.config(text="Haar cascade file missing.", fg="red")
            return
        faceCascade = cv2.CascadeClassifier(haarcasecade_path)

        # optional label_map (Label -> Enrollment)
        label_map_inv = {}
        try:
            if os.path.exists(label_map_path):
                with open(label_map_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            lbl = int(row.get("Label", -1))
                            enroll = str(row.get("Enrollment", "")).strip()
                            if enroll:
                                label_map_inv[lbl] = enroll
                        except Exception:
                            pass
        except Exception:
            pass

        # camera
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cam.isOpened():
            message_label.config(text="Cannot open camera", fg="red")
            try:
                text_to_speech("Cannot open camera. Check device permissions.")
            except Exception:
                pass
            return

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # warm-up
        for _ in range(5):
            cam.read()

        font_cv = cv2.FONT_HERSHEY_SIMPLEX
        attendance = pd.DataFrame(columns=["Enrollment", "Name"], dtype=str)
        debug_lines = []
        start = time.time()
        duration = 20
        end_time = start + duration

        try:
            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

                for (x, y, w, h) in faces:
                    try:
                        predicted_label, conf = recognizer.predict(gray[y:y + h, x:x + w])
                    except Exception:
                        predicted_label, conf = None, 100.0

                    dbg = f"raw:{predicted_label} conf:{int(conf)}"
                    cv2.putText(frame, dbg, (x, y + h + 18), font_cv, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

                    recognized_enroll = None
                    if predicted_label is not None and conf < 70:
                        # try label_map first
                        recognized_enroll = label_map_inv.get(predicted_label)
                        # fallback: numeric label as enrollment string or partial match
                        if recognized_enroll is None:
                            s_label = str(predicted_label)
                            if s_label in student_map:
                                recognized_enroll = s_label
                            else:
                                # partial match search
                                for k in student_map.keys():
                                    if s_label in str(k):
                                        recognized_enroll = k
                                        break

                    # if mapping produced a NAME instead of enrollment (trainer mistake), try to map name -> enrollment
                    if recognized_enroll is not None and recognized_enroll not in student_map:
                        # maybe it's a name — try to find enrollment by name
                        candidates = [en for en, nm in student_map.items() if str(nm).strip().lower() == str(recognized_enroll).strip().lower()]
                        if candidates:
                            recognized_enroll = candidates[0]
                        else:
                            # search contains
                            candidates = [en for en, nm in student_map.items() if str(recognized_enroll).strip().lower() in str(nm).strip().lower()]
                            if candidates:
                                recognized_enroll = candidates[0]
                            else:
                                # cannot map
                                recognized_enroll = None

                    if recognized_enroll is not None:
                        name = student_map.get(str(recognized_enroll), "")
                        # append safely (avoid array-like values)
                        attendance.loc[len(attendance)] = [str(recognized_enroll), str(name)]
                        label_text = f"{recognized_enroll} - {name}"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, label_text, (x, y - 8), font_cv, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                        debug_lines.append(f"OK:{recognized_enroll}/{int(conf)}")
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y - 8), font_cv, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        debug_lines.append(f"UNK/{int(conf)}")

                    if len(debug_lines) > 6:
                        debug_lines = debug_lines[-6:]

                # drop duplicates by Enrollment
                if not attendance.empty:
                    attendance = attendance.drop_duplicates(subset=["Enrollment"], keep="first")

                # overlay debug
                for idx, ln in enumerate(debug_lines):
                    cv2.putText(frame, ln, (8, 20 + idx * 18), font_cv, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
                if debug_lines:
                    try:
                        message_label.config(text=debug_lines[-1], fg="yellow")
                    except Exception:
                        pass

                cv2.imshow("Filling Attendance...", frame)
                if cv2.waitKey(30) & 0xFF == 27:
                    break
                if time.time() > end_time:
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()

        # Save attendance CSV: Enrollment, Name, <date> columns
        try:
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H-%M-%S")
            folder = os.path.join(attendance_path, Subject)
            os.makedirs(folder, exist_ok=True)
            fileName = os.path.join(folder, f"{Subject}_{date}_{timeStamp}.csv")

            if attendance.empty:
                # create an empty file with headers if none recognized
                empty_df = pd.DataFrame(columns=["Enrollment", "Name"])
                empty_df[date] = 0
                empty_df.to_csv(fileName, index=False)
            else:
                attendance = attendance.drop_duplicates(subset=["Enrollment"], keep="first")
                attendance[date] = 1
                attendance["Name"] = attendance["Name"].astype(str).str.replace(r"[\[\]]", "", regex=True).str.strip()
                attendance.to_csv(fileName, index=False)

            message_label.config(text=f"Attendance saved: {fileName}", fg="green")
            try:
                text_to_speech(f"Attendance filled successfully for {Subject}")
            except Exception:
                pass
        except Exception as exc:
            message_label.config(text=f"Failed saving attendance: {exc}", fg="red")
            return

        # show result window with Email option
        try:
            rv = tk.Toplevel(win)
            rv.title(f"Attendance of {Subject}")
            rv.configure(background="#0b0e17")
            with open(fileName, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for r_idx, row in enumerate(reader):
                    for c_idx, val in enumerate(row):
                        lbl = tk.Label(rv, text=str(val), bg="#0b0e17", fg="#e6eef8", font=("Segoe UI", 10), bd=1, relief=tk.RIDGE, padx=8, pady=4)
                        lbl.grid(row=r_idx, column=c_idx, sticky="nsew")

            def on_email_click():
                recipient = faculty_email_var.get().strip()
                if not recipient:
                    recipient = simpledialog.askstring("Faculty Email", "Enter faculty email:", parent=rv)
                    if not recipient:
                        messagebox.showinfo("Cancelled", "No recipient provided")
                        return
                creds = _interactive_smtp_dialog(rv)
                if not creds or not creds[0]:
                    messagebox.showinfo("Cancelled", "No SMTP credentials provided")
                    return
                sender, pwd, srv, port = creds
                ok, err = _send_email_with_attachment(sender, pwd, srv, port, recipient, f"Attendance: {Subject}", f"Attendance for {Subject}", fileName)
                if ok:
                    messagebox.showinfo("Email sent", f"Attendance emailed to {recipient}")
                else:
                    messagebox.showerror("Email failed", f"Failed to send email: {err}")

            tk.Button(rv, text="Email to Faculty", command=on_email_click, bg="#7ad3ff", fg="#04242a", bd=0).grid(row=0, column=10, padx=8, pady=8, sticky="ne")
            rv.transient(win)
            rv.grab_set()
        except Exception as exc:
            message_label.config(text=f"Saved but cannot show result window: {exc}", fg="orange")

    # Manage students popup (list + per-row delete)
    def open_manage_students_ui():
        df = _read_students_df()
        popup = tk.Toplevel(win)
        popup.title("Manage Registered Students")
        popup.geometry("760x420")
        popup.configure(bg="#0b0e17")
        popup.transient(win)
        popup.grab_set()

        header = tk.Frame(popup, bg="#0b0e17", pady=8)
        header.pack(fill="x")
        tk.Label(header, text="Registered students", fg="#e6eef8", bg="#0b0e17", font=("Segoe UI Semibold", 14)).pack(side="left", padx=12)
        info_label = tk.Label(header, text=f"{len(df)} students", fg="#a8b3c7", bg="#0b0e17", font=("Segoe UI", 10))
        info_label.pack(side="right", padx=12)

        # scrollable area
        container = tk.Frame(popup, bg="#0b0e17")
        container.pack(fill="both", expand=True, padx=12, pady=(0,12))
        canvas = tk.Canvas(container, bg="#0b0e17", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg="#0b0e17")
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # headers
        hdr = tk.Frame(scrollable, bg="#0b0e17")
        hdr.pack(fill="x", pady=(4,6))
        tk.Label(hdr, text="Enrollment", width=18, anchor="w", fg="#e6eef8", bg="#0b0e17", font=("Segoe UI Semibold", 10)).grid(row=0, column=0, padx=(6,6))
        tk.Label(hdr, text="Name", width=36, anchor="w", fg="#e6eef8", bg="#0b0e17", font=("Segoe UI Semibold", 10)).grid(row=0, column=1, padx=(6,6))
        tk.Label(hdr, text="RegisteredOn", width=30, anchor="w", fg="#e6eef8", bg="#0b0e17", font=("Segoe UI Semibold", 10)).grid(row=0, column=2, padx=(6,6))
        tk.Label(hdr, text="Action", width=12, anchor="center", fg="#e6eef8", bg="#0b0e17", font=("Segoe UI Semibold", 10)).grid(row=0, column=3, padx=(6,6))

        rows_frames = []

        def make_delete_callback(enroll, rowf):
            def _delete():
                confirm = messagebox.askyesno("Confirm", f"Delete student {enroll}? This will remove CSV entry and training images.")
                if not confirm:
                    return
                ok, err = _remove_student_from_csv(enroll)
                if not ok:
                    messagebox.showerror("Failed", f"Failed to remove CSV: {err}")
                    return
                removed = _delete_training_images_for(enroll)
                # optionally remove trained model — ask user
                if os.path.exists(trainimagelabel_path):
                    rm = messagebox.askyesno("Model removal", "Do you also want to remove the trained model file? (You will need to retrain)")
                    if rm:
                        try:
                            bak = trainimagelabel_path + f".bak.{int(time.time())}"
                            shutil.copyfile(trainimagelabel_path, bak)
                        except Exception:
                            pass
                        try:
                            os.remove(trainimagelabel_path)
                        except Exception:
                            pass
                # remove row from UI
                try:
                    rowf.destroy()
                except Exception:
                    pass
                messagebox.showinfo("Deleted", f"Deleted {enroll}. Removed {removed} images.")
            return _delete

        for _, r in df.iterrows():
            rf = tk.Frame(scrollable, bg="#0b0e17", pady=6)
            rf.pack(fill="x", padx=2)
            rows_frames.append(rf)
            e_lbl = tk.Label(rf, text=str(r["Enrollment"]), width=18, anchor="w", fg="#e6eef8", bg="#0b0e17", font=("Segoe UI", 10))
            e_lbl.grid(row=0, column=0, padx=(6,6))
            n_lbl = tk.Label(rf, text=str(r["Name"]), width=36, anchor="w", fg="#e6eef8", bg="#0b0e17", font=("Segoe UI", 10))
            n_lbl.grid(row=0, column=1, padx=(6,6))
            t_lbl = tk.Label(rf, text=str(r["RegisteredOn"]), width=30, anchor="w", fg="#a8b3c7", bg="#0b0e17", font=("Segoe UI", 10))
            t_lbl.grid(row=0, column=2, padx=(6,6))

            del_btn = tk.Button(rf, text="Delete", width=10, bg="#7ad3ff", fg="#04242a", bd=0, command=make_delete_callback(str(r["Enrollment"]), rf))
            del_btn.grid(row=0, column=3, padx=(12,8))

        if df.empty:
            tk.Label(scrollable, text="No registered students found.", fg="#a8b3c7", bg="#0b0e17", font=("Segoe UI", 11)).pack(pady=12)

        footer = tk.Frame(popup, bg="#0b0e17")
        footer.pack(fill="x", pady=(6,12))
        tk.Button(footer, text="Close", command=popup.destroy, bg="#111418", fg="yellow", bd=0, padx=12, pady=6).pack(side="right", padx=12)

    # Buttons row
    check_btn = tk.Button(btn_frame, text="Check Sheets", command=lambda: _open_folder(os.path.join(attendance_path, subject_var.get().strip())), bg="#111418", fg="yellow", font=("Segoe UI", 11), bd=0, padx=12, pady=8)
    check_btn.grid(row=0, column=0, padx=(0,12))

    fill_btn = tk.Button(btn_frame, text="Fill Attendance (Camera)", command=FillAttendance_camera, bg=ACCENT, fg="#04242a", font=("Segoe UI Semibold", 11), bd=0, padx=12, pady=8)
    fill_btn.grid(row=0, column=1, padx=(0,12))

    manage_btn = tk.Button(btn_frame, text="Manage Students", command=open_manage_students_ui, bg="#7ad3ff", fg="#04242a", font=("Segoe UI", 11), bd=0, padx=12, pady=8)
    manage_btn.grid(row=0, column=2, padx=(0,12))

    close_btn = tk.Button(btn_frame, text="Exit", command=win.destroy, bg="#111418", fg="yellow", font=("Segoe UI", 11), bd=0, padx=12, pady=8)
    close_btn.grid(row=0, column=3)

    footer = tk.Label(card, text="Attendance directory: " + os.path.abspath(attendance_path), fg="#a8b3c7", bg=CARD_BG, font=("Segoe UI", 8))
    footer.grid(row=4, column=0, columnspan=2, sticky="w", pady=(12,0))

    entry.focus_set()
