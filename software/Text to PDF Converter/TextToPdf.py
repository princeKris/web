import argparse
import threading
import itertools
import sys
import os
from fpdf import FPDF
import time


# -------------------------------------------------
# Helper to load bundled files when using PyInstaller
# -------------------------------------------------
def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller exe.
    """
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return relative_path


# Spinner function
def spinner(text="Processing"):
    for char in itertools.cycle("|/-\\"):
        if stop_spinner:
            break
        sys.stdout.write(f"\r{text}... {char}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\rDone!                \n")

# main function
def main():

    # -------------------------------------------------
    # CLI for the App
    # -------------------------------------------------
    parser = argparse.ArgumentParser(
        description="""Convert a TXT file to PDF.
        example > py TextToPdf.py -i 'textfilename.txt' -o 'outputpdfname.pdf'
        for extra options like fontsize, border, align and line space check out below.""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-i", "--input", help="Input text file path (with extension)")
    parser.add_argument("-o", "--output", default="output.pdf",help="Output PDF filename (default: output.pdf)")
    parser.add_argument("-s", "--fontsize", type=int, default=14,help="Font size (default: 14)")
    parser.add_argument("-l", "--linespace", type=int, default=6,help="Line spacing (default: 6)")
    parser.add_argument("-b", "--border", type=int, default=0,help="Border (0=no border, 1=full border, default: 0)")
    parser.add_argument("-a", "--align", choices=["L", "C", "R", "J"], default="L",help="Text alignment: L, C, R, J (default: L)")
    args = parser.parse_args()

    # --- INPUT for App if there is no args value it will ask the user ---
    if args.input:
        file_path = args.input
        output_name = args.output
        font_size = args.fontsize
        line_space = args.linespace
        border = args.border
        align = args.align
    else:
        print(" TXT To PDF ".center(40, "="))
        file_path = input("Enter input file path: ").strip()
        if not file_path:
            print("Error: Input file path is required.")
            sys.exit()
        output_name = input("Enter output PDF name (default output.pdf): ").strip() or "output.pdf"
        font_size = int(input("Font size (default 14): ").strip() or 14)
        line_space = int(input("Line spacing (default 6): ").strip() or 6)
        border = int(input("Border (0 or 1) default 0: ").strip() or 0)
        align = input("Align (L,C,R,J default L): ").strip().upper() or "L"

    # -------------------
    # PDF SETUP
    # -------------------
    pdf = FPDF()
    pdf.add_page()

    # Load font safely
    font_path = resource_path("arial-unicode-ms.ttf")
    pdf.add_font("arialuni", "", font_path)
    pdf.set_font("arialuni", "", font_size)

    # Read text file
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit()

    # Start spinner thread
    global stop_spinner
    stop_spinner = False
    t = threading.Thread(target=spinner, args=("Generating PDF",))
    t.start()

    # Generate the PDF
    pdf.multi_cell(0, line_space, content, border=border, align=align)
    pdf.output(output_name)

    # Stop spinner
    stop_spinner = True
    t.join()

    print(f"\nPDF created successfully: {output_name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"Error: {err}")
