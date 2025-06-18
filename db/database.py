import sqlite3
import os
import pandas as pd
from create import create_images_table
from create import create_users_table

default_pandas = True

class Database: 
    def __init__(self, db_path = None):
        """
        Initializes the Database class with the specified database name.
        If no path is provided, it defaults to "db/mma.db".
        """
        if db_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(script_dir, "mma.db")

        self.db_path = db_path

        # create database + tables if it does not yet exist
        if not os.path.exists(self.db_path):
            self.connect()
            self.reset_tables()
            self.close()

        self.connection = None
        self.cursor = None

    def connect(self):
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            self.cursor = None

    def execute_query(self, query, params=None):
        if not self.connection:
            raise Exception("Database not connected.")
        if params is None:
            params = ()
        self.cursor.execute(query, params)
        self.connection.commit()
        return self.cursor.fetchall()
    
    def fetch_dataframe(self, query, params=None):
        """
        Executes a query and returns the result as a pandas DataFrame.
        """
        if not self.connection:
            raise Exception("Database not connected.")
        if params is None:
            params = ()
        df = pd.read_sql_query(query, self.connection, params=params)
        return df

    def fetch_image_by_id(self, id, pandas=default_pandas):
        query = "SELECT * FROM images WHERE id = ?"
        if pandas:
            return self.fetch_dataframe(query, (id,))
        return self.execute_query(query, (id,))
    
    def fetch_image_by_path(self, image_path, pandas=default_pandas):
        query = "SELECT * FROM images WHERE image_path = ?"
        if pandas:
            return self.fetch_dataframe(query, (image_path,))
        return self.execute_query(query, (image_path,))

    def fetch_all_images(self, pandas=default_pandas):
        query = "SELECT * FROM images"
        if pandas:
            return self.fetch_dataframe(query)
        return self.execute_query(query)
    
    def fetch_all_user_images(self, user_id, pandas=default_pandas):
        query = "SELECT * FROM images WHERE user_id = ?"
        if pandas:
            return self.fetch_dataframe(query, (user_id,))
        return self.execute_query(query, (user_id,))
    
    def fetch_user_by_id(self, user_id, pandas=default_pandas):
        query = "SELECT * FROM users WHERE user_id = ?"
        if pandas:
            return self.fetch_dataframe(query, (user_id,))
        return self.execute_query(query, (user_id,))
    
    def fetch_user_by_username(self, username, pandas=default_pandas):
        query = "SELECT * FROM users WHERE username = ?"
        if pandas:
            return self.fetch_dataframe(query, (username,))
        return self.execute_query(query, (username,))
    
    def fetch_all_users(self, pandas=default_pandas):
        query = "SELECT * FROM users"
        if pandas:
            return self.fetch_dataframe(query)
        return self.execute_query(query)
    
    def insert_image(self, user_id: int, image_path: str, guidance: int, strength: float, 
                    prompt: str, prompt_number: int, enhanced_prompt: bool = False, selected: bool = False):
        """
        Insert an image record into the database.
        
        Args:
            user_id (int): User ID
            image_path (str): Path to the image file
            guidance (float): Guidance value between 0-10
            strength (float): Strength value between 0-1
            prompt (str): Text prompt
            prompt_number (int): Prompt number
            enhanced_prompt (bool): Whether prompt is enhanced (default: False)
            selected (bool): Whether image is selected (default: False)
        """
        # Validate arguments
        if not isinstance(user_id, int):
            raise ValueError("user_id must be an integer")
        if not isinstance(image_path, str):
            raise ValueError("image_path must be a string")
        if not isinstance(guidance, (int, float)) or not (0 <= guidance <= 10):
            raise ValueError("guidance must be a float between 0-10")
        if not isinstance(strength, (int, float)) or not (0 <= strength <= 1):
            raise ValueError("strength must be a float between 0-1")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if not isinstance(prompt_number, int):
            raise ValueError("prompt_number must be an integer")
        if not isinstance(enhanced_prompt, bool):
            raise ValueError("enhanced_prompt must be a boolean")
        if not isinstance(selected, bool):
            raise ValueError("selected must be a boolean")
        
        query = """
        INSERT INTO images (user_id, image_path, guidance, strength, prompt, prompt_number, enhanced_prompt, selected)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (user_id, image_path, guidance, strength, prompt, prompt_number, enhanced_prompt, selected)
        try:
            self.execute_query(query, params)
        except sqlite3.IntegrityError as e:
            print(f"Error inserting image: {e}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def insert_user(self, username):
        """
        Insert an image record into the database.
        
        Args:
            username (str): Username to insert into the users table.
        """
        if not isinstance(username, str):
            raise ValueError("username must be a string")

        query = """
        INSERT INTO users (username)
        VALUES (?)
        """
        params = (username,)
        try:
            self.execute_query(query, params)
        except sqlite3.IntegrityError as e:
            print(f"Error inserting user: {e}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def insert_user_if_not_exists(self, username):
        """
        Inserts a user into the database if they do not already exist.

        Args:
            username (str): Username to insert into the users table.
        """
        if not isinstance(username, str):
            raise ValueError("username must be a string")

        query = "INSERT OR IGNORE INTO users (username) VALUES (?)"
        params = (username,)
        try:
            self.execute_query(query, params)
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def reset_images(self):
        """
        Resets the images table by dropping it if it exists and recreating it.
        """
        self.execute_query("DROP TABLE IF EXISTS images;")
        self.execute_query(create_images_table)

    def reset_users(self):
        """
        Resets the users table by dropping it if it exists and recreating it.
        """
        self.execute_query("DROP TABLE IF EXISTS users;")
        self.execute_query(create_users_table)

    def reset_tables(self):
        """
        Resets both the images and users tables.
        """
        self.reset_images()
        self.reset_users()

    def __enter__(self):
        """
        Allows the Database class to be used as a context manager.
        Automatically connects to the database when entering the context.
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Ensures the database connection is closed when exiting the context.
        """
        self.close()