"""
Download the required NLTK data.
"""
import nltk

def main():
    print("Downloading NLTK data...")
    
    nltk.download('punkt')
    nltk.download('punkt_tab')
    print("NLTK data downloaded successfully.")


if __name__ == "__main__":
    main()
