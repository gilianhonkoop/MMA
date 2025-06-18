import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../db')))
from database import Database

print("Import works!")

with Database() as db:
        df = db.fetch_all_images()
        print(db.fetch_all_users(pandas=True))