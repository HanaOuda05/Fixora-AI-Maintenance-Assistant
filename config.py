from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MANUALS_DIR = BASE_DIR / "data" / "manuals"
PROCESSED_DIR = BASE_DIR / "data_processed"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

MANUALS = {
    "sc6002xl": {
        "manufacturer": "Siemens",
        "device": "SC 6002XL Patient Monitor",
        "file": MANUALS_DIR / "sc6002xl_patient_monitor.pdf",
    },

    "servo_ventilator": {
        "manufacturer": "Siemens",
        "device": "Servo Ventilator System",
        "file": MANUALS_DIR / "servo_ventilator_service_manual.pdf",
    },

    "philips_g40": {
        "manufacturer": "Philips",
        "device": "G30/G40 Patient Monitor",
        "file": MANUALS_DIR / "philips_g30_g40_patient_monitor.pdf",
    },
}