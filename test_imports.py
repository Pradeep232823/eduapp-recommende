libraries = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scikit-learn", "sklearn"),
    ("tensorflow", "tensorflow"),
    ("flask", "flask"),
    ("google-play-scraper", "google_play_scraper"),
    ("scipy", "scipy"),
    ("gunicorn", "gunicorn"),
    ("psycopg2-binary", "psycopg2")
]

print("Checking required libraries...\n")

for lib_name, import_name in libraries:
    try:
        __import__(import_name)
        print(f"{lib_name} imported successfully")
    except Exception as e:
        print(f"{lib_name} import failed → {e}")

print("\nLibrary verification completed.")