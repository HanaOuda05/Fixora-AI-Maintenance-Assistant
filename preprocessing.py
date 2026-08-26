import pymupdf
import re
import pdfplumber
import json 

from config import MANUALS, PROCESSED_DIR




def clean_text(text):
    """
    Basic cleanup for extracted PDF text.
    """
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)

    return text.strip() 

def clean_cell(value):
    if value is None:
        return ""

    value = value.replace("\n", " ")

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    value = re.sub(
        r"\bpage(\d)",
        r"page \1",
        value,
    )

    value = re.sub(
        r"\.See\b",
        ". See",
        value,
    )

    return value.strip()

def fix_medical_terms(text):

    if not text:
        return text

    text = re.sub(
        r"\bSpO\s+2\b",
        "SpO2",
        text,
    )

    text = re.sub(
        r"\betCO\s+2\b",
        "etCO2",
        text,
    )

    text = re.sub(
        r"\bCO\s+2\b",
        "CO2",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()

def extract_pdf_text(pdf_path):
    document = pymupdf.open(pdf_path)
    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")

        pages.append(
            {
                "page": page_number,
                "text": text.strip(),
            }
        )

    document.close()

    return pages

def test_manuals():

    for manual_id, manual in MANUALS.items():

        print("=" * 70)

        print(f"Manual ID: {manual_id}")
        print(f"Device: {manual['device']}")
        print(f"Manufacturer: {manual['manufacturer']}")
        print(f"File: {manual['file']}")

        print("-" * 70)

        pages = extract_pdf_text(
            manual["file"]
        )

        print(f"Total pages: {len(pages)}")

        non_empty_pages = [
            page
            for page in pages
            if page["text"]
        ]

        print(
            f"Pages with extracted text: "
            f"{len(non_empty_pages)}"
        )

        if non_empty_pages:

            print("\nSample from first extracted page:\n")

            print(
                non_empty_pages[0]["text"][:1000]
            )

        print()

def extract_servo_error_chunks(pages, manual):

    chunks = []

    inside_error_table = False

    for page in pages:

        page_number = page["page"]
        raw_text = page["text"]

        if not raw_text:
            continue

        text = clean_text(raw_text)

        # Start only when the real table is found
        if (
            "Technical error codes" in text
            and "Error code" in text
            and "Error message / Possible cause" in text
            and "Recommended action" in text
        ):
            inside_error_table = True

        if not inside_error_table:
            continue

        # Stop when preventive maintenance begins
        if (
            "Preventive maintenance" in text
            and "Technical error codes" not in text
        ):
            break

        pattern = r"""
        (?<!\d)
        (\d{1,5})
        \s+
        ([A-Z][A-Z0-9_ ]+?)
        (?=
            \s+\d+\.
            |
            \s+N/A
            |
            \s+\d{1,5}\s+[A-Z]
            |
            $
        )
        """

        matches = list(
            re.finditer(
                pattern,
                text,
                re.VERBOSE,
            )
        )

        for index, match in enumerate(matches):

            error_code = match.group(1).strip()

            # Reject document code noise
            if error_code == "382":
                continue

            error_message = (
                match.group(2)
                .strip()
                .replace("  ", " ")
            )
            error_message = re.sub( r"_\s+", "_",error_message,)

            start = match.end()

            if index + 1 < len(matches):
                end = matches[index + 1].start()
            else:
                end = len(text)

            action_text = text[start:end].strip()

            action_text = re.sub(
                r"^Recommended action\s*",
                "",
                action_text,
                flags=re.IGNORECASE,
            )

            chunk_text = (
                f"Error code: {error_code}. "
                f"Error message / possible cause: "
                f"{error_message}. "
                f"Recommended action: "
                f"{action_text if action_text else 'Not specified.'}"
            )

            chunks.append(
                {
                    "device_id": "servo_ventilator",
                    "device": manual["device"],
                    "manufacturer": manual["manufacturer"],
                    "page": page_number,
                    "section": "Technical error codes",
                    "chunk_type": "error_code",
                    "error_code": error_code,
                    "error_message": error_message,
                    "recommended_action": action_text,
                    "text": chunk_text,
                }
            )

    return chunks

def test_servo_error_chunks():

    manual = MANUALS["servo_ventilator"]

    pages = extract_pdf_text(
        manual["file"]
    )

    chunks = extract_servo_error_chunks(
        pages,
        manual,
    )

    print("=" * 70)
    print("SERVO ERROR CODE TEST")
    print("=" * 70)

    print(
        f"Total error-code chunks: "
        f"{len(chunks)}"
    )

    print()

    for chunk in chunks[:10]:

        print("-" * 70)

        print(
            f"Error code: "
            f"{chunk['error_code']}"
        )

        print(
            f"Page: "
            f"{chunk['page']}"
        )

        print(
            f"Message: "
            f"{chunk['error_message']}"
        )

        print(
            f"Action: "
            f"{chunk['recommended_action']}"
        )

        print()

    print("\nCHECK ERROR CODE 37")
    print("=" * 70)

    for chunk in chunks:
      if chunk["error_code"] == "37":
        print(chunk)    

def extract_philips_troubleshooting_chunks(manual):

    chunks = []

    current_symptom = None

    with pdfplumber.open(manual["file"]) as pdf:

        # Troubleshooting pages
        for page_number in range(45, 51):

            page = pdf.pages[page_number - 1]

            tables = page.extract_tables()

            for table_number, table in enumerate(
                tables,
                start=1
            ):

                if not table:
                    continue

                current_symptom = None

                # -----------------------------------
                # Determine section
                # -----------------------------------

                section = "Troubleshooting"

                if page_number in [45, 46]:
                    section = "Power Problems"

                elif page_number == 47:
                    section = "Display Problems"

                elif page_number == 48:

                    if table_number == 1:
                        section = "Alarm Problems"

                    elif table_number == 2:
                        section = "NIBP Problems"

                elif page_number == 49:

                    if table_number == 1:
                        section = "NIBP Problems"

                    elif table_number == 2:
                        section = "Temperature Problems"

                elif page_number == 50:

                    if table_number == 1:
                        section = "SpO2 Problems"

                    elif table_number == 2:
                        section = "etCO2 Problems"

                    elif table_number == 3:
                        section = "C.O. Problems"

                # -----------------------------------
                # Process rows
                # -----------------------------------

                for row in table:

                    if not row:
                        continue

                    if len(row) < 3:
                        continue

                    symptom = clean_cell(row[0])
                    cause = clean_cell(row[1])
                    action = clean_cell(row[2])

                    # Ignore header
                    if (
                        symptom.lower() == "symptom"
                        and "possible cause"
                        in cause.lower()
                    ):
                        continue

                    # New symptom
                    if symptom:
                        current_symptom = symptom

                    # Blank symptom = same symptom
                    if not current_symptom:
                        continue

                    if not cause:
                        continue

                    if not action:
                        continue

                    # -----------------------------------
                    # Clean common PDF extraction issues
                    # -----------------------------------

                    symptom = fix_medical_terms(
                        current_symptom
                    )

                    cause = fix_medical_terms(
                        cause
                    )

                    action = fix_medical_terms(
                        action
                    )

                    chunk_text = (
                        f"Symptom: {symptom}. "
                        f"Possible cause: {cause}. "
                        f"Action: {action}"
                    )

                    chunks.append(
                        {
                            "device_id":
                                "philips_g40",

                            "device":
                                manual["device"],

                            "manufacturer":
                                manual["manufacturer"],

                            "page":
                                page_number,

                            "section":
                                section,

                            "chunk_type":
                                "troubleshooting",

                            "error_code":
                                None,

                            "symptom":
                                symptom,

                            "possible_cause":
                                cause,

                            "action":
                                action,

                            "text":
                                chunk_text,
                        }
                    )

    return chunks



def test_philips_troubleshooting_chunks():

    manual = MANUALS[
        "philips_g40"
    ]

    chunks = (
        extract_philips_troubleshooting_chunks(
            manual
        )
    )

    print("=" * 70)

    print(
        "PHILIPS TROUBLESHOOTING TEST"
    )

    print("=" * 70)

    print(
        f"Total troubleshooting chunks: "
        f"{len(chunks)}"
    )

    print()

    for chunk in chunks[:20]:

        print("-" * 70)

        print(
            f"Page: "
            f"{chunk['page']}"
        )

        print(
            f"Section: "
            f"{chunk['section']}"
        )

        print(
            f"Symptom: "
            f"{chunk['symptom']}"
        )

        print(
            f"Possible cause: "
            f"{chunk['possible_cause']}"
        )

        print(
            f"Action: "
            f"{chunk['action']}"
        )

        print()


def extract_sc6002xl_troubleshooting_chunks(manual):

    chunks = []

    with pdfplumber.open(manual["file"]) as pdf:

        # Troubleshooting pages
        for page_number in range(73, 79):

            page = pdf.pages[page_number - 1]

            tables = page.extract_tables()

            for table_number, table in enumerate(
                tables,
                start=1,
            ):

                if not table:
                    continue

                # ------------------------------
                # Determine section
                # ------------------------------

                section = "Troubleshooting"

                if page_number in [73, 74]:
                    section = "Power Problems"

                elif page_number == 75:

                    section_map = {
                        1: "Power-off Alarm Malfunction",
                        2: "Power-up Process Malfunction",
                        3: "Rotary Knob Malfunction",
                        4: "LCD Display Malfunction",
                    }

                    section = section_map.get(
                        table_number,
                        "Troubleshooting"
                    )

                elif page_number == 76:

                    section_map = {
                        1: "LCD Display Malfunction",
                        2: "Fixed Key Malfunction",
                        3: "Alarm Malfunctions",
                    }

                    section = section_map.get(
                        table_number,
                        "Troubleshooting"
                    )

                elif page_number == 77:

                    section_map = {
                        1: "NBP Malfunctions",
                        2: "etCO2 Malfunctions",
                    }

                    section = section_map.get(
                        table_number,
                        "Troubleshooting"
                    )

                elif page_number == 78:
                    section = "Recorder Malfunctions"

                # ------------------------------
                # Process rows
                # ------------------------------

                for row in table:

                    if not row:
                        continue

                    if len(row) < 3:
                        continue

                    symptom = clean_cell(row[0])
                    cause = clean_cell(row[1])
                    action = clean_cell(row[2])

                    # Ignore empty / image-only rows
                    if not symptom:
                        continue

                    if not cause:
                        continue

                    if not action:
                        continue

                    # Ignore header row
                    symptom_lower = symptom.lower()

                    if (
                        symptom_lower
                        in [
                            "conditions",
                            "symptom(s)",
                            "symptoms",
                        ]
                        and
                        "possible cause"
                        in cause.lower()
                    ):
                        continue

                    symptom = fix_medical_terms(
                        symptom
                    )

                    cause = fix_medical_terms(
                        cause
                    )

                    action = fix_medical_terms(
                        action
                    )

                    chunk_text = (
                        f"Symptom or condition: "
                        f"{symptom}. "
                        f"Possible cause: "
                        f"{cause}. "
                        f"Troubleshooting and remedial action: "
                        f"{action}"
                    )

                    chunks.append(
                        {
                            "device_id":
                                "sc6002xl",

                            "device":
                                manual["device"],

                            "manufacturer":
                                manual["manufacturer"],

                            "page":
                                page_number,

                            "section":
                                section,

                            "chunk_type":
                                "troubleshooting",

                            "error_code":
                                None,

                            "symptom":
                                symptom,

                            "possible_cause":
                                cause,

                            "action":
                                action,

                            "text":
                                chunk_text,
                        }
                    )

    return chunks

def test_sc6002xl_troubleshooting_chunks():

    manual = MANUALS["sc6002xl"]

    chunks = (
        extract_sc6002xl_troubleshooting_chunks(
            manual
        )
    )

    print("=" * 70)

    print(
        "SC6002XL TROUBLESHOOTING TEST"
    )

    print("=" * 70)

    print(
        f"Total troubleshooting chunks: "
        f"{len(chunks)}"
    )

    print()

    for chunk in chunks[:20]:

        print("-" * 70)

        print(
            f"Page: "
            f"{chunk['page']}"
        )

        print(
            f"Section: "
            f"{chunk['section']}"
        )

        print(
            f"Symptom: "
            f"{chunk['symptom']}"
        )

        print(
            f"Possible cause: "
            f"{chunk['possible_cause']}"
        )

        print(
            f"Action: "
            f"{chunk['action']}"
        )

        print()

def test_sc6002xl_tables():

    manual = MANUALS["sc6002xl"]

    with pdfplumber.open(
        manual["file"]
    ) as pdf:

        for page_number in range(73, 79):

            page = pdf.pages[
                page_number - 1
            ]

            print("=" * 80)
            print(
                f"PAGE {page_number}"
            )
            print("=" * 80)

            tables = page.extract_tables()

            print(
                f"Number of tables found: "
                f"{len(tables)}"
            )

            for table_number, table in enumerate(
                tables,
                start=1,
            ):

                print()
                print(
                    f"TABLE {table_number}"
                )
                print("-" * 80)

                for row in table:
                    print(row)

            print()



def build_all_chunks():

    all_chunks = []

    # =====================================
    # 1. Servo
    # =====================================

    servo_manual = MANUALS[
        "servo_ventilator"
    ]

    servo_pages = extract_pdf_text(
        servo_manual["file"]
    )

    servo_chunks = extract_servo_error_chunks(
        servo_pages,
        servo_manual,
    )

    all_chunks.extend(
        servo_chunks
    )


    # =====================================
    # 2. Philips G30/G40
    # =====================================

    philips_manual = MANUALS[
        "philips_g40"
    ]

    philips_chunks = (
        extract_philips_troubleshooting_chunks(
            philips_manual
        )
    )

    all_chunks.extend(
        philips_chunks
    )


    # =====================================
    # 3. SC6002XL
    # =====================================

    sc_manual = MANUALS[
        "sc6002xl"
    ]

    sc_chunks = (
        extract_sc6002xl_troubleshooting_chunks(
            sc_manual
        )
    )

    all_chunks.extend(
        sc_chunks
    )


    # =====================================
    # Add unique IDs
    # =====================================

    for index, chunk in enumerate(
        all_chunks,
        start=1,
    ):

        chunk["chunk_id"] = (
            f"chunk_{index:04d}"
        )


    return all_chunks


def save_chunks_to_json(chunks):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        PROCESSED_DIR
        / "maintai_chunks.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("=" * 70)

    print(
        f"Saved {len(chunks)} chunks"
    )

    print(
        f"Output file: {output_file}"
    )

    print("=" * 70)

def validate_chunks(chunks):

    print("=" * 70)
    print("CHUNK VALIDATION")
    print("=" * 70)

    required_fields = [
        "chunk_id",
        "device_id",
        "device",
        "manufacturer",
        "page",
        "section",
        "chunk_type",
        "text",
    ]

    missing_field_count = 0
    empty_text_count = 0
    duplicate_ids = 0

    seen_ids = set()

    for chunk in chunks:

        # --------------------------
        # Check required fields
        # --------------------------

        for field in required_fields:

            if field not in chunk:

                print(
                    f"Missing field '{field}' "
                    f"in chunk: {chunk}"
                )

                missing_field_count += 1


        # --------------------------
        # Check text
        # --------------------------

        if not chunk.get("text", "").strip():

            print(
                f"Empty text in: "
                f"{chunk.get('chunk_id')}"
            )

            empty_text_count += 1


        # --------------------------
        # Check duplicate IDs
        # --------------------------

        chunk_id = chunk.get(
            "chunk_id"
        )

        if chunk_id in seen_ids:

            print(
                f"Duplicate chunk ID: "
                f"{chunk_id}"
            )

            duplicate_ids += 1

        seen_ids.add(
            chunk_id
        )


    print()

    print(
        f"Missing fields: "
        f"{missing_field_count}"
    )

    print(
        f"Empty texts: "
        f"{empty_text_count}"
    )

    print(
        f"Duplicate IDs: "
        f"{duplicate_ids}"
    )

    print()

    if (
        missing_field_count == 0
        and empty_text_count == 0
        and duplicate_ids == 0
    ):

        print(
            "Validation passed."
        )

    else:

        print(
            "Validation found problems."
        )

    print("=" * 70)    
def run_preprocessing():

    print("=" * 70)
    print("MAINTAI PREPROCESSING")
    print("=" * 70)

    chunks = build_all_chunks()

    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )

    print()

    servo_count = sum(
        1
        for chunk in chunks
        if chunk["device_id"]
        == "servo_ventilator"
    )

    philips_count = sum(
        1
        for chunk in chunks
        if chunk["device_id"]
        == "philips_g40"
    )

    sc_count = sum(
        1
        for chunk in chunks
        if chunk["device_id"]
        == "sc6002xl"
    )

    print(
        f"Servo chunks: {servo_count}"
    )

    print(
        f"Philips chunks: {philips_count}"
    )

    print(
        f"SC6002XL chunks: {sc_count}"
    )

    print()

    validate_chunks(
        chunks
    )

    save_chunks_to_json(
        chunks
    )
if __name__ == "__main__":

    # Development tests
    # test_manuals()
    # test_servo_error_chunks()
    # test_philips_tables()
    # test_philips_troubleshooting_chunks()
    # test_sc6002xl_tables()
    # test_sc6002xl_troubleshooting_chunks()

    # Actual preprocessing pipeline
    run_preprocessing()