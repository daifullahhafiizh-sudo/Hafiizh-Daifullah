import time
import cv2
import serial
from ultralytics import YOLO

# ============================================================
# KONFIGURASI MODEL
# ============================================================
MODEL_PATH = "best.pt"

IMG_SIZE = 640
CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45

# ============================================================
# KONFIGURASI WEBCAM
# ============================================================
CAMERA_INDEX = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ============================================================
# KONFIGURASI ESP32
# ============================================================
SERIAL_ENABLED = True
SERIAL_PORT = "COM8"
BAUD_RATE = 115200

# ============================================================
# VALIDASI YOLO
# ============================================================
REQUIRED_FRAMES = 5

# Jeda minimum command yang sama
COMMAND_DELAY = 2.0

# ============================================================
# VALIDASI TOF
# ============================================================
# Rentang orang yang dianggap berada di zona akses.
# 300-1200 mm = 30 cm sampai 1,2 m.
TOF_MIN_DISTANCE_MM = 1000
TOF_MAX_DISTANCE_MM = 2000

# Minimal jumlah pembacaan TOF valid yang diterima Python
# sebelum OPEN diizinkan.
TOF_REQUIRED_SAMPLES = 3

# Jika tidak menerima data TOF selama waktu ini,
# Python menganggap status TOF tidak valid.
TOF_TIMEOUT = 1.0

# ============================================================
# PANEL
# ============================================================
PANEL_ALPHA = 0.35

# ============================================================
# CLASS MODEL
# ============================================================
CLASS_PERSON = "Orang"
CLASS_HELMET = "Helmet"
CLASS_GLOVE = "Sarung Tangan"
CLASS_VEST = "Rompi"

# ============================================================
# WARNA BGR
# ============================================================
COLOR_PERSON = (255, 0, 0)
COLOR_HELMET = (0, 255, 0)
COLOR_GLOVE = (255, 0, 255)
COLOR_VEST = (0, 255, 255)

COLOR_COMPLETE = (0, 255, 0)
COLOR_INCOMPLETE = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_WARNING = (0, 165, 255)


def normalize_class(name):
    return str(name).strip().lower()


def load_model(path):
    print("Loading model...")
    try:
        model = YOLO(path)
        print("Model berhasil dimuat.")
        return model
    except Exception as e:
        print("\nERROR: Model gagal dimuat.")
        print(e)
        raise SystemExit


def check_required_classes(model, required_classes):
    model_class_names = [normalize_class(n) for n in model.names.values()]

    print("\nMemeriksa class model...")
    for required_class in required_classes:
        if required_class in model_class_names:
            print(f"[OK] {required_class}")
        else:
            print(f"[WARNING] Class {required_class} tidak ditemukan!")


def connect_esp32(port, baud):
    if not SERIAL_ENABLED:
        return None

    try:
        conn = serial.Serial(port, baud, timeout=0.05)
        time.sleep(2)

        # Bersihkan pesan startup ESP32
        conn.reset_input_buffer()
        print(f"\nESP32 terhubung: {port}")
        return conn

    except Exception as e:
        print("\nWARNING: ESP32 tidak terhubung.")
        print(e)
        print("Deteksi tetap berjalan tanpa kontrol portal.")
        return None


class CommandSender:
    def __init__(self, esp32, min_delay):
        self.esp32 = esp32
        self.min_delay = min_delay
        self.last_command = ""
        self.last_command_time = 0

    def send(self, command):
        now = time.time()

        if (command == self.last_command
                and (now - self.last_command_time) < self.min_delay):
            return

        if self.esp32 is not None:
            try:
                self.esp32.write((command + "\n").encode())
                print("ESP32 <-", command)
            except Exception as e:
                print("Serial error:", e)

        self.last_command = command
        self.last_command_time = now


class FpsCounter:
    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self.prev_time = None

    def tick(self):
        now = time.time()

        if self.prev_time is not None:
            elapsed = now - self.prev_time
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                self.fps = (
                    self.smoothing * self.fps
                    + (1 - self.smoothing) * instant_fps
                )

        self.prev_time = now
        return self.fps


def read_esp32_messages(esp32, tof_state):
    if esp32 is None:
        return tof_state

    try:
        while esp32.in_waiting:
            line = esp32.readline().decode(errors="ignore").strip()

            if not line:
                continue

            # Format ESP32:
            # TOF,distance_mm,valid
            if line.startswith("TOF,"):
                parts = line.split(",")

                if len(parts) >= 3:
                    try:
                        distance = int(parts[1])
                        valid = parts[2] == "1"
                        now = time.time()

                        # Validasi kedua di sisi Python
                        valid = (
                            valid
                            and TOF_MIN_DISTANCE_MM <= distance <= TOF_MAX_DISTANCE_MM
                        )

                        if valid:
                            tof_state["valid_samples"] += 1
                        else:
                            tof_state["valid_samples"] = 0

                        tof_state["distance_mm"] = distance
                        tof_state["valid"] = (
                            valid
                            and tof_state["valid_samples"] >= TOF_REQUIRED_SAMPLES
                        )
                        tof_state["last_update"] = now

                    except ValueError:
                        pass

            else:
                # Tampilkan pesan debug ESP32
                print("ESP32 ->", line)

    except Exception as e:
        print("Error membaca ESP32:", e)

    # Timeout komunikasi TOF
    if time.time() - tof_state["last_update"] > TOF_TIMEOUT:
        tof_state["valid"] = False
        tof_state["valid_samples"] = 0

    return tof_state


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_detections(frame, detections, names):
    person_detected = False
    helmet_detected = False
    glove_detected = False
    vest_detected = False

    for det in detections:
        class_name = det["class_normalized"]
        confidence = det["confidence"]
        x1, y1, x2, y2 = det["box"]

        if class_name == names["person"]:
            person_detected = True
            color = COLOR_PERSON
        elif class_name == names["helmet"]:
            helmet_detected = True
            color = COLOR_HELMET
        elif class_name == names["glove"]:
            glove_detected = True
            color = COLOR_GLOVE
        elif class_name == names["vest"]:
            vest_detected = True
            color = COLOR_VEST
        else:
            color = COLOR_WHITE

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{det['class']} {confidence:.2f}"
        cv2.putText(
            frame, label, (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    return person_detected, helmet_detected, glove_detected, vest_detected


def draw_info_panel(frame, flags, current_status, portal_status, fps, tof_state):
    person_detected, helmet_detected, glove_detected, vest_detected = flags

    panel_x1, panel_y1 = 10, 10
    panel_x2, panel_y2 = 560, 300

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0, dst=frame)

    rows = [
        (
            f"{'Orang':<13}: {'TERDETEKSI' if person_detected else 'TIDAK'}",
            COLOR_PERSON if person_detected else COLOR_WHITE,
            40
        ),
        (
            f"{'Helmet':<13}: {'OK' if helmet_detected else 'NO'}",
            COLOR_HELMET if helmet_detected else COLOR_INCOMPLETE,
            70
        ),
        (
            f"{'Rompi':<13}: {'OK' if vest_detected else 'NO'}",
            COLOR_VEST if vest_detected else COLOR_INCOMPLETE,
            100
        ),
        (
            f"{'Sarung Tangan':<13}: {'OK' if glove_detected else 'NO'}",
            COLOR_GLOVE if glove_detected else COLOR_INCOMPLETE,
            130
        ),
    ]

    for text, color, y in rows:
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # --------------------------------------------------------
    # STATUS TOF
    # --------------------------------------------------------
    distance = tof_state["distance_mm"]

    if tof_state["valid"]:
        tof_text = f"TOF: {distance} mm (ZONA VALID)"
        tof_color = COLOR_COMPLETE
    elif distance > 0:
        tof_text = f"TOF: {distance} mm (TIDAK VALID)"
        tof_color = COLOR_WARNING
    else:
        tof_text = "TOF: TIDAK ADA DATA"
        tof_color = COLOR_INCOMPLETE

    cv2.putText(frame, tof_text, (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tof_color, 2)

    # --------------------------------------------------------
    # STATUS APD
    # --------------------------------------------------------
    if current_status == "APD LENGKAP":
        status_color = COLOR_COMPLETE
    elif current_status == "APD TIDAK LENGKAP":
        status_color = COLOR_INCOMPLETE
    else:
        status_color = COLOR_WHITE

    cv2.putText(frame, current_status, (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    # --------------------------------------------------------
    # STATUS PORTAL
    # --------------------------------------------------------
    if portal_status == "AKSES DITERIMA":
        portal_color = COLOR_COMPLETE
    elif portal_status == "AKSES DITOLAK":
        portal_color = COLOR_INCOMPLETE
    else:
        portal_color = COLOR_WHITE

    cv2.putText(frame, portal_status, (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.65, portal_color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2)


def main():
    print("==============================================")
    print(" SISTEM DETEKSI APD YOLOv5n + TOF400C")
    print("==============================================\n")

    model = load_model(MODEL_PATH)

    print("\nClass yang terdapat pada model:")
    for class_id, class_name in model.names.items():
        print(f"{class_id} = {class_name}")

    names = {
        "person": normalize_class(CLASS_PERSON),
        "helmet": normalize_class(CLASS_HELMET),
        "glove": normalize_class(CLASS_GLOVE),
        "vest": normalize_class(CLASS_VEST),
    }

    check_required_classes(model, list(names.values()))

    esp32 = connect_esp32(SERIAL_PORT, BAUD_RATE)
    sender = CommandSender(esp32, COMMAND_DELAY)

    tof_state = {
        "distance_mm": 0,
        "valid": False,
        "valid_samples": 0,
        "last_update": time.time(),
    }

    print("\nMembuka webcam eksternal...")
    cap = open_camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)

    if cap is None:
        print("\nERROR: Webcam eksternal tidak dapat dibuka.")
        if esp32 is not None:
            esp32.close()
        raise SystemExit

    print("\nWebcam eksternal berhasil.")
    print("Camera Index:", CAMERA_INDEX)
    print(
        "Resolusi:",
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "x",
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print("\nTekan Q untuk keluar.")

    complete_counter = 0
    incomplete_counter = 0
    no_person_counter = 0

    current_status = "MENUNGGU"
    portal_status = "MENUNGGU"

    fps_counter = FpsCounter()

    try:
        while True:
            # ------------------------------------------------
            # BACA DATA TOF DARI ESP32
            # ------------------------------------------------
            tof_state = read_esp32_messages(esp32, tof_state)

            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca webcam.")
                break

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------
            results = model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
            )
            result = results[0]

            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = str(model.names[class_id])

                    detections.append({
                        "class": class_name,
                        "class_normalized": normalize_class(class_name),
                        "confidence": confidence,
                        "box": (x1, y1, x2, y2),
                    })

            flags = draw_detections(frame, detections, names)
            person_detected, helmet_detected, glove_detected, vest_detected = flags

            # ------------------------------------------------
            # LOGIKA APD
            # ------------------------------------------------
            if person_detected:
                if helmet_detected and glove_detected and vest_detected:
                    complete_counter += 1
                    incomplete_counter = 0
                    no_person_counter = 0
                    current_status = "APD LENGKAP"
                else:
                    incomplete_counter += 1
                    complete_counter = 0
                    no_person_counter = 0
                    current_status = "APD TIDAK LENGKAP"
            else:
                no_person_counter += 1
                complete_counter = 0
                incomplete_counter = 0
                current_status = "TIDAK ADA ORANG"

            # ------------------------------------------------
            # KEPUTUSAN AKSES
            # ------------------------------------------------
            if complete_counter >= REQUIRED_FRAMES:
                # ==================================================
                # PENTING:
                # APD lengkap SAJA tidak cukup.
                # TOF harus valid.
                # ==================================================
                if tof_state["valid"]:
                    portal_status = "AKSES DITERIMA"
                    sender.send("OPEN")
                else:
                    # Foto yang hanya terdeteksi YOLO tetapi
                    # tidak berada pada zona TOF tidak boleh
                    # membuka portal.
                    portal_status = "TOF TIDAK VALID"

                complete_counter = REQUIRED_FRAMES

            elif incomplete_counter >= REQUIRED_FRAMES:
                portal_status = "AKSES DITOLAK"
                sender.send("CLOSE")
                incomplete_counter = REQUIRED_FRAMES

            elif no_person_counter >= REQUIRED_FRAMES:
                portal_status = "MENUNGGU ORANG"
                sender.send("NO_PERSON")
                no_person_counter = REQUIRED_FRAMES

            fps = fps_counter.tick()
            draw_info_panel(frame, flags, current_status, portal_status, fps, tof_state)

            cv2.imshow("YOLOv5n + TOF400C - DETEKSI APD", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if esp32 is not None:
            esp32.close()
        print("\nProgram selesai.")


if __name__ == "__main__":
    main()