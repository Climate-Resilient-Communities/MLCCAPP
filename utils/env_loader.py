from dotenv import load_dotenv

def load_environment():
    """
    Loads environment variables from the .env file.
    Call this function wherever you need to ensure environment variables are loaded.
    """
    load_dotenv()