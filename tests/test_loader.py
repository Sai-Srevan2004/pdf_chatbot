# tests/test_loader.py
from app.loader import load_and_chunk

def test_load_and_chunk():
    chunks = load_and_chunk("SaiSrevan_Resume.pdf")
    for doc in chunks:
        print("\n","page:",doc.page_content)
    print(f"Loaded {len(chunks)} chunks")

if __name__ == "__main__":
    test_load_and_chunk()