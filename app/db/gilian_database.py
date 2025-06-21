import sqlite3
import os
import json
import pandas as pd
from modules.prompt import Prompt
from modules.prompt_image import PromptImage
from modules.chat import Chat
from .create import create_images_table
from .create import create_users_table
from .create import create_chats_table
from .create import create_prompts_table

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

    # Image method: Handles retrieving image metadata.
    def fetch_image_by_id(self, id, pandas=default_pandas):
        query = "SELECT * FROM images WHERE id = ?"
        if pandas:
            return self.fetch_dataframe(query, (id,))
        return self.execute_query(query, (id,))
    
    # Image method: Handles retrieving image metadata.
    def fetch_image_by_path(self, path, pandas=default_pandas):
        query = "SELECT * FROM images WHERE path = ?"
        if pandas:
            return self.fetch_dataframe(query, (path,))
        return self.execute_query(query, (path,))

    # Image method: Handles retrieving image metadata.
    def fetch_all_images(self, pandas=default_pandas):
        query = "SELECT * FROM images"
        if pandas:
            return self.fetch_dataframe(query)
        return self.execute_query(query)
    
    # Image method: Handles retrieving image metadata.
    def fetch_images_by_user(self, user_id, pandas=default_pandas, include_depth=False, is_selected=None):
        """
        Fetches all images associated with a specific user.
        Args:
            user_id (int): The ID of the user to fetch images for.
            pandas (bool): If True, returns a pandas DataFrame; otherwise, returns a list of tuples.
            include_depth (bool): If True, includes depth information in the result.
            is_selected (bool or None): If True, fetch only selected images; if False, fetch only unselected images; if None, fetch all images.
        Returns:
            A pandas DataFrame or a list of tuples containing image metadata.
        """
        if not isinstance(user_id, int):
            raise ValueError("user_id must be an integer")

        if include_depth:
            query = """
            SELECT images.*, prompts.depth
            FROM images
            LEFT JOIN prompts ON images.input_prompt_id = prompts.id
            WHERE images.user_id = ?
            """
        else:
            query = "SELECT * FROM images WHERE user_id = ?"

        # Add selected filter if specified
        if is_selected is not None:
            query += " AND images.selected = ?"
            params = (user_id, 1 if is_selected else 0)
        else:
            params = (user_id,)

        if pandas:
            return self.fetch_dataframe(query, params)
        return self.execute_query(query, params)

    # Image method: Handles retrieving image metadata.    
    def fetch_images_by_chat(self, chat_id, pandas=default_pandas, include_depth=False):
        """
        Fetches all images associated with a specific chat.
        Args:
            chat_id (int): The ID of the chat to fetch images for.
            pandas (bool): If True, returns a pandas DataFrame; otherwise, returns a list of tuples.
            include_depth (bool): If True, includes depth information in the result.
        Returns:
            A pandas DataFrame or a list of tuples containing image metadata.
        """
        if not isinstance(chat_id, int):
            raise ValueError("chat_id must be an integer")
        if include_depth:
            query = """
            SELECT images.*, prompts.depth
            FROM images
            LEFT JOIN prompts ON images.input_prompt_id = prompts.id
            WHERE images.chat_id = ?
            """
        else:
            query = "SELECT * FROM images WHERE chat_id = ?"
        if pandas:
            return self.fetch_dataframe(query, (chat_id,))
        return self.execute_query(query, (chat_id,))

    # Image method: Handles retrieving image metadata.    
    def fetch_images_by_prompt(self, prompt_id, pandas=default_pandas):
        query = "SELECT * FROM images WHERE prompt_id = ?"
        if pandas:
            return self.fetch_dataframe(query, (prompt_id,))
        return self.execute_query(query, (prompt_id,))
    
    # User method: Supports user lookup.
    def fetch_user_by_id(self, id, pandas=default_pandas):
        query = "SELECT * FROM users WHERE id = ?"
        if pandas:
            return self.fetch_dataframe(query, (id,))
        return self.execute_query(query, (id,))
   
    # User method: Supports user lookup.
    def fetch_user_by_username(self, username, pandas=default_pandas):
        query = "SELECT * FROM users WHERE username = ?"
        if pandas:
            return self.fetch_dataframe(query, (username,))
        return self.execute_query(query, (username,))
    
    # User method: Supports user lookup.
    def fetch_all_users(self, pandas=default_pandas):
        query = "SELECT * FROM users"
        if pandas:
            return self.fetch_dataframe(query)
        return self.execute_query(query)

    # Chat and prompt methods: Allow storing and querying hierarchical data: user → chat → prompt.
    # Chat method: handles retrieving chat    
    def fetch_chat_by_id(self, id, pandas=default_pandas):
        query = "SELECT * FROM chats WHERE id = ?"
        if pandas:
            return self.fetch_dataframe(query, (id,))
        return self.execute_query(query, (id,))

    # Chat method: handles retrieving chat    
    def fetch_chats_by_user(self, user_id, pandas=default_pandas):
        query = "SELECT * FROM chats WHERE user_id = ?"
        if pandas:
            return self.fetch_dataframe(query, (user_id,))
        return self.execute_query(query, (user_id,))

    # Chat method: handles retrieving chats    
    def fetch_all_chats(self, pandas=default_pandas):
        query = "SELECT * FROM chats"
        if pandas:
            return self.fetch_dataframe(query)
        return self.execute_query(query)

    # Prompt method: handles retrieving prompt        
    def fetch_prompt_by_id(self, id, pandas=default_pandas):
        query = "SELECT * FROM prompts WHERE id = ?"
        if pandas:
            return self.fetch_dataframe(query, (id,))
        return self.execute_query(query, (id,))

    # Prompt method: handles retrieving prompt            
    def fetch_prompts_by_chat(self, chat_id, pandas=default_pandas):
        query = "SELECT * FROM prompts WHERE chat_id = ?"
        if pandas:
            return self.fetch_dataframe(query, (chat_id,))
        return self.execute_query(query, (chat_id,))

    # Prompt method: handles retrieving prompt            
    def fetch_prompts_by_user(self, user_id, pandas=default_pandas):
        query = "SELECT * FROM prompts WHERE user_id = ?"
        if pandas:
            return self.fetch_dataframe(query, (user_id,))
        return self.execute_query(query, (user_id,))

    # Prompt method: handles retrieving prompt            
    def fetch_all_prompts(self, pandas=default_pandas):
        query = "SELECT * FROM prompts"
        if pandas:
            return self.fetch_dataframe(query)
        return self.execute_query(query)
    
    # Image method: Handles adding image metadata.
    # input_prompt_id and output_prompt_id connect images to generation prompts.
    def insert_image(self, id: str, user_id: int, chat_id: int, prompt_guidance: float, 
                    image_guidance: float, path: str, input_prompt_id: str, output_prompt_id: str,):
        """
        Insert an image record into the database.
        
        Args:
            id (str): Image ID
            user_id (int): User ID
            chat_id (int): Chat ID
            prompt_guidance (float): Prompt guidance value
            image_guidance (float): Image guidance value
            path (str): Path to the image file
            input_prompt_id (str): ID of the input prompt
            output_prompt_id (str): ID of the output prompt
        """
        # Validate arguments
        if not isinstance(id, str):
            raise ValueError("id must be a string")
        if not isinstance(input_prompt_id, str):
            raise ValueError("input_prompt_id must be an string")
        if not isinstance(output_prompt_id, str):
            raise ValueError("output_prompt_id must be an string")
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
        INSERT INTO images (id, input_prompt_id, output_prompt_id, user_id, chat_id, prompt_guidance, image_guidance, path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (id, input_prompt_id, output_prompt_id, user_id, chat_id, prompt_guidance, image_guidance, path)
        try:
            self.execute_query(query, params)
        except sqlite3.IntegrityError as e:
            print(f"Error inserting image: {e}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    # User method: Supports user creation
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

    # User method: Supports user creation
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

    # Chat method: Supports adding chat data
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

        return self.cursor.lastrowid

    def insert_prompt(self, id, chat_id, user_id, prompt, depth, used_suggestion=False, 
                     modified_suggestion=False, suggestion_used=None, is_enhanced=False, 
                     enhanced_prompt=None, image_in_id=None, images_out=None):
        """
        Insert a prompt record into the database.
        
        Args:
            id (str): prompt ID
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
        if not isinstance(id, str):
            raise ValueError("id must be a string")
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
        # if image_in_id is not None and not isinstance(image_in_id, int):
        #     raise ValueError("image_in_id must be an integer or None")
        # if images_out is not None and not isinstance(images_out, str):
        #     raise ValueError("images_out must be a string or None")

        query = """
        INSERT INTO prompts (id, chat_id, user_id, prompt, depth, used_suggestion, modified_suggestion,
                           suggestion_used, is_enhanced, enhanced_prompt, image_in_id, images_out)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (id, chat_id, user_id, prompt, depth, used_suggestion, modified_suggestion,
                 suggestion_used, is_enhanced, enhanced_prompt, image_in_id, images_out)
        try:
            self.execute_query(query, params)
        except sqlite3.IntegrityError as e:
            print(f"Error inserting prompt: {e}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    # Reset methods
    # Drops and recreates all tables using the SQL from create.py. 
    # Used when reinitializing your database.
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

    # Context manager support
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

    def save_image(self, prompt_image: PromptImage, session_id: str, user_id: int = 1):
        if not isinstance(prompt_image, PromptImage):
            raise TypeError("Expected prompt_image to be an instance of PromptImage")
        
        self.connect()
        
        try:
            self.insert_image(
                id=prompt_image.id,
                user_id=user_id,
                chat_id=session_id,
                prompt_guidance=prompt_image.prompt_guidance if prompt_image.prompt_guidance else 0.0,
                image_guidance=prompt_image.image_guidance if prompt_image.image_guidance else 0.0,
                path=prompt_image.path,
                input_prompt_id=prompt_image.input_prompt,
                output_prompt_id=prompt_image.output_prompt
            )
            return True
        except Exception as e:
            print(f"Error saving image to database: {e}")
            return False
        finally:
            self.close()

    def save_prompt(self, prompt: Prompt, chat_id: str, user_id: int = 1):
        if not isinstance(prompt, Prompt):
            raise TypeError("Expected prompt to be an instance of Prompt")
        
        self.connect()
        
        try:
            images_out_json = json.dumps([img.id for img in prompt.images_out]) if prompt.images_out else None
            
            input_image_id = prompt.input_image.id if prompt.input_image else None
            
            self.insert_prompt(
                id=prompt.id,
                chat_id=chat_id,
                user_id=user_id,
                prompt=prompt.prompt,
                depth=prompt.depth,
                used_suggestion=prompt.used_suggestion,
                modified_suggestion=prompt.modified_suggestion,
                suggestion_used=prompt.suggestion_used,
                is_enhanced=prompt.is_enhanced,
                enhanced_prompt=prompt.enhanced_prompt,
                image_in_id=input_image_id,
                images_out=images_out_json
            )
            return True
        except Exception as e:
            print(f"Error saving prompt to database: {e}")
            return False
        finally:
            self.close()

    def save_chat(self, chat: Chat):
        if not isinstance(chat, Chat):
            raise TypeError("Expected chat to be an instance of Chat")
        
        self.connect()
        
        try:
            chat_id = self.insert_chat(chat.title, chat.user_id)
            
            for prompt in chat.prompts:
                self.save_prompt(prompt, chat_id, chat.user_id)
            
            return chat_id
        except Exception as e:
            print(f"Error saving chat to database: {e}")
            return None
        finally:
            self.close()
    
    def get_image_by_id(self, image_id):
        if image_id is None:
            return None
        
        self.connect()
        
        try:
            result = self.fetch_image_by_id(image_id, pandas=False)
            if not result or not result[0]:
                return None
            
            record = result[0]
            id_val, input_prompt_id, output_prompt_id, user_id, chat_id, prompt_guidance, image_guidance, path, *_ = record
            
            if os.path.exists(path):
                from PIL import Image
                image = Image.open(path)
                
                from modules.prompt_image import PromptImage
                prompt_image = PromptImage(image, prompt_guidance, image_guidance, 
                                        input_prompt=input_prompt_id, 
                                        output_prompt=output_prompt_id,
                                        save=False)
                
                prompt_image.id = id_val
                
                prompt_image.path = path
                
                return prompt_image
            else:
                print(f"Image file not found at path: {path}")
                return None
        except Exception as e:
            print(f"Error getting image from database: {e}")
            return None
        finally:
            self.close()
    
    def set_image_selected(self, image_id, selected=True):
        if image_id is None:
            return False
        
        self.connect()
        
        try:
            query = "UPDATE images SET selected = ? WHERE id = ?"
            self.execute_query(query, (1 if selected else 0, image_id))
            return True
        except Exception as e:
            print(f"Error setting image selected status: {e}")
            return False
        finally:
            self.close()

    def delete_image(self, image_id):
        try:
            query = "DELETE FROM images WHERE id = ?"
            self.execute_query(query, (image_id,))
            return True
        except Exception as e:
            print(f"Error deleting image from database: {e}")
            return False