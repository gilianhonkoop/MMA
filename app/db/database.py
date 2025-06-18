import sqlite3
import os
import pandas as pd
from create import create_images_table
from create import create_users_table
from create import create_chats_table
from create import create_prompts_table

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
    
    def fetch_image_by_path(self, path, pandas=default_pandas):
        query = "SELECT * FROM images WHERE path = ?"
        if pandas:
            return self.fetch_dataframe(query, (path,))
        return self.execute_query(query, (path,))

    def fetch_all_images(self, pandas=default_pandas):
        query = "SELECT * FROM images"
        if pandas:
            return self.fetch_dataframe(query)
        return self.execute_query(query)
    
    def fetch_images_by_user(self, user_id, pandas=default_pandas):
        query = "SELECT * FROM images WHERE user_id = ?"
        if pandas:
            return self.fetch_dataframe(query, (user_id,))
        return self.execute_query(query, (user_id,))
        
    def fetch_images_by_chat(self, chat_id, pandas=default_pandas):
        query = "SELECT * FROM images WHERE chat_id = ?"
        if pandas:
            return self.fetch_dataframe(query, (chat_id,))
        return self.execute_query(query, (chat_id,))
        
    def fetch_images_by_prompt(self, prompt_id, pandas=default_pandas):
        query = "SELECT * FROM images WHERE prompt_id = ?"
        if pandas:
            return self.fetch_dataframe(query, (prompt_id,))
        return self.execute_query(query, (prompt_id,))
    
    def fetch_user_by_id(self, id, pandas=default_pandas):
        query = "SELECT * FROM users WHERE id = ?"
        if pandas:
            return self.fetch_dataframe(query, (id,))
        return self.execute_query(query, (id,))
    
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
        
    def fetch_chat_by_id(self, id, pandas=default_pandas):
        query = "SELECT * FROM chats WHERE id = ?"
        if pandas:
            return self.fetch_dataframe(query, (id,))
        return self.execute_query(query, (id,))
        
    def fetch_chats_by_user(self, user_id, pandas=default_pandas):
        query = "SELECT * FROM chats WHERE user_id = ?"
        if pandas:
            return self.fetch_dataframe(query, (user_id,))
        return self.execute_query(query, (user_id,))
        
    def fetch_all_chats(self, pandas=default_pandas):
        query = "SELECT * FROM chats"
        if pandas:
            return self.fetch_dataframe(query)
        return self.execute_query(query)
        
    def fetch_prompt_by_id(self, id, pandas=default_pandas):
        query = "SELECT * FROM prompts WHERE id = ?"
        if pandas:
            return self.fetch_dataframe(query, (id,))
        return self.execute_query(query, (id,))
        
    def fetch_prompts_by_chat(self, chat_id, pandas=default_pandas):
        query = "SELECT * FROM prompts WHERE chat_id = ?"
        if pandas:
            return self.fetch_dataframe(query, (chat_id,))
        return self.execute_query(query, (chat_id,))
        
    def fetch_prompts_by_user(self, user_id, pandas=default_pandas):
        query = "SELECT * FROM prompts WHERE user_id = ?"
        if pandas:
            return self.fetch_dataframe(query, (user_id,))
        return self.execute_query(query, (user_id,))
        
    def fetch_all_prompts(self, pandas=default_pandas):
        query = "SELECT * FROM prompts"
        if pandas:
            return self.fetch_dataframe(query)
        return self.execute_query(query)
    
    def insert_image(self, prompt_id: int, user_id: int, chat_id: int, prompt_guidance: float, 
                    image_guidance: float, path: str):
        """
        Insert an image record into the database.
        
        Args:
            prompt_id (int): Prompt ID
            user_id (int): User ID
            chat_id (int): Chat ID
            prompt_guidance (float): Prompt guidance value
            image_guidance (float): Image guidance value
            path (str): Path to the image file
        """
        # Validate arguments
        if not isinstance(prompt_id, int):
            raise ValueError("prompt_id must be an integer")
        if not isinstance(user_id, int):
            raise ValueError("user_id must be an integer")
        if not isinstance(chat_id, int):
            raise ValueError("chat_id must be an integer")
        if not isinstance(prompt_guidance, (int, float)):
            raise ValueError("prompt_guidance must be a float")
        if not isinstance(image_guidance, (int, float)):
            raise ValueError("image_guidance must be a float")
        if not isinstance(path, str):
            raise ValueError("path must be a string")
        
        query = """
        INSERT INTO images (prompt_id, user_id, chat_id, prompt_guidance, image_guidance, path)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (prompt_id, user_id, chat_id, prompt_guidance, image_guidance, path)
        try:
            self.execute_query(query, params)
        except sqlite3.IntegrityError as e:
            print(f"Error inserting image: {e}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def insert_user(self, username, password):
        """
        Insert an user record into the database.
        
        Args:
            username (str): Username to insert into the users table.
            password (str): Password for the user.
        """
        if not isinstance(username, str):
            raise ValueError("username must be a string")
        if not isinstance(password, str):
            raise ValueError("password must be a string")

        query = """
        INSERT INTO users (username, password)
        VALUES (?, ?)
        """
        params = (username, password)
        try:
            self.execute_query(query, params)
        except sqlite3.IntegrityError as e:
            print(f"Error inserting user: {e}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def insert_user_if_not_exists(self, username, password="default_password"):
        """
        Inserts a user into the database if they do not already exist.

        Args:
            username (str): Username to insert into the users table.
            password (str): Password for the user (default: "default_password").
        """
        if not isinstance(username, str):
            raise ValueError("username must be a string")
        if not isinstance(password, str):
            raise ValueError("password must be a string")

        query = "INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)"
        params = (username, password)
        try:
            self.execute_query(query, params)
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def insert_chat(self, title, user_id):
        """
        Insert a chat record into the database.
        
        Args:
            title (str): Chat title
            user_id (int): User ID
        """
        if not isinstance(title, str):
            raise ValueError("title must be a string")
        if not isinstance(user_id, int):
            raise ValueError("user_id must be an integer")

        query = """
        INSERT INTO chats (title, user_id)
        VALUES (?, ?)
        """
        params = (title, user_id)
        try:
            self.execute_query(query, params)
        except sqlite3.IntegrityError as e:
            print(f"Error inserting chat: {e}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def insert_prompt(self, chat_id, user_id, prompt, depth, used_suggestion=False, 
                     modified_suggestion=False, suggestion_used=None, is_enhanced=False, 
                     enhanced_prompt=None, image_in_id=None, images_out=None):
        """
        Insert a prompt record into the database.
        
        Args:
            chat_id (int): Chat ID
            user_id (int): User ID
            prompt (str): User prompt text
            depth (int): Depth in the conversation
            used_suggestion (bool): Whether a suggestion was used (default: False)
            modified_suggestion (bool): Whether the suggestion was modified (default: False)
            suggestion_used (str): The suggestion text that was used (default: None)
            is_enhanced (bool): Whether the prompt was enhanced (default: False)
            enhanced_prompt (str): The enhanced prompt text (default: None)
            image_in_id (int): ID of the input image (default: None)
            images_out (str): Comma-separated list of output image IDs (default: None)
        """
        if not isinstance(chat_id, int):
            raise ValueError("chat_id must be an integer")
        if not isinstance(user_id, int):
            raise ValueError("user_id must be an integer")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if not isinstance(depth, int):
            raise ValueError("depth must be an integer")
        if not isinstance(used_suggestion, bool):
            raise ValueError("used_suggestion must be a boolean")
        if not isinstance(modified_suggestion, bool):
            raise ValueError("modified_suggestion must be a boolean")
        if suggestion_used is not None and not isinstance(suggestion_used, str):
            raise ValueError("suggestion_used must be a string or None")
        if not isinstance(is_enhanced, bool):
            raise ValueError("is_enhanced must be a boolean")
        if enhanced_prompt is not None and not isinstance(enhanced_prompt, str):
            raise ValueError("enhanced_prompt must be a string or None")
        if image_in_id is not None and not isinstance(image_in_id, int):
            raise ValueError("image_in_id must be an integer or None")
        if images_out is not None and not isinstance(images_out, str):
            raise ValueError("images_out must be a string or None")

        query = """
        INSERT INTO prompts (chat_id, user_id, prompt, depth, used_suggestion, modified_suggestion,
                           suggestion_used, is_enhanced, enhanced_prompt, image_in_id, images_out)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (chat_id, user_id, prompt, depth, used_suggestion, modified_suggestion,
                 suggestion_used, is_enhanced, enhanced_prompt, image_in_id, images_out)
        try:
            self.execute_query(query, params)
        except sqlite3.IntegrityError as e:
            print(f"Error inserting prompt: {e}")
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
        
    def reset_chats(self):
        """
        Resets the chats table by dropping it if it exists and recreating it.
        """
        self.execute_query("DROP TABLE IF EXISTS chats;")
        self.execute_query(create_chats_table)
        
    def reset_prompts(self):
        """
        Resets the prompts table by dropping it if it exists and recreating it.
        """
        self.execute_query("DROP TABLE IF EXISTS prompts;")
        self.execute_query(create_prompts_table)

    def reset_tables(self):
        """
        Resets all tables.
        """
        self.reset_users()
        self.reset_chats()
        self.reset_prompts()
        self.reset_images()

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