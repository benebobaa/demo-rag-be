import os
import shutil
from dotenv import load_dotenv

load_dotenv()

from app.database import engine, Base
from app.models_db import DocumentModel, Session, Message, GraphTriple

def reset_database():
    print(f"🗑️  Dropping all tables from: {engine.url}")
    Base.metadata.drop_all(bind=engine)
    
    print("✨  Recreating tables...")
    Base.metadata.create_all(bind=engine)
    
    print("✅  Database schema reset.")

def clear_uploads():
    upload_dir = "uploads"
    if os.path.exists(upload_dir):
        print(f"🗑️  Clearing {upload_dir} directory...")
        shutil.rmtree(upload_dir)
        os.makedirs(upload_dir, exist_ok=True)
        print("✅  Uploads cleared.")
    else:
        print(f"ℹ️  {upload_dir} directory not found.")
        os.makedirs(upload_dir, exist_ok=True)

if __name__ == "__main__":
    confirm = input("⚠️  WARNING: This will DELETE ALL DATA in the connected database and uploads. Continue? (y/n): ")
    if confirm.lower() == 'y':
        reset_database()
        clear_uploads()
        print("\n🚀 System reset complete. Restart the backend to ensure a clean state.")
    else:
        print("❌ Action cancelled.")
