import mysql.connector
from mysql.connector import Error
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging for this module
logger = logging.getLogger(__name__)
# Ensure basicConfig is called, ideally in the main application entry point,
# but can be here if this module might be used standalone.
# logging.basicConfig(level=logging.INFO) # Already configured in agent.py, might be redundant or configured by application

class AppointmentDB:
    def __init__(self): # Removed default args, will fetch from env
        try:
            db_host = os.getenv('MYSQL_HOST', 'localhost')
            db_user = os.getenv('MYSQL_USER', 'root')
            db_password = os.getenv('MYSQL_PASSWORD', 'root') # Defaulting to 'root' as per original logic
            db_name = os.getenv('MYSQL_DATABASE_NAME', 'Appointments')
            db_port_str = os.getenv('MYSQL_PORT', '3306')

            # Log warnings if defaults are used for critical params
            if db_host == 'localhost':
                logger.warning("MYSQL_HOST not set, using default 'localhost'.")
            if db_user == 'root':
                logger.warning("MYSQL_USER not set, using default 'root'.")
            # Not logging warning for default password for security reasons (avoiding log noise if 'root' is intentional for dev)
            if db_name == 'Appointments':
                logger.warning("MYSQL_DATABASE_NAME not set, using default 'Appointments'.")
            if db_port_str == '3306':
                logger.info("MYSQL_PORT not set, using default '3306'.") # Info, as 3306 is very standard

            try:
                db_port = int(db_port_str)
            except ValueError:
                logger.warning(f"Invalid MYSQL_PORT value '{db_port_str}', using default 3306.")
                db_port = 3306

            logger.info(f"Attempting DB connection with: host='{db_host}', user='{db_user}', database='{db_name}', port={db_port}")

            self.conn = mysql.connector.connect(
                host=db_host,
                database=db_name,
                user=db_user,
                password=db_password,
                port=db_port
            )
            self._setup_database()
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}") # Use logger
            raise
        
    def _setup_database(self):
        """Create the appointments table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS appointments (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                date DATE NOT NULL,
                time TIME NOT NULL,
                purpose TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
        cursor.close()
    
    def book_appointment(self, name: str, date: str, time: str, purpose: str) -> str:
        """Book a new appointment."""
        try:
            # Validate date format
            datetime.strptime(date, '%Y-%m-%d')
            # Validate time format
            datetime.strptime(time, '%H:%M')
            
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO appointments (name, date, time, purpose)
                VALUES (%s, %s, %s, %s)
            ''', (name, date, time, purpose))
            self.conn.commit()
            # cursor.close() # Closing in finally
            
            return f"Successfully booked appointment for {name} on {date} at {time}"
        except ValueError as e:
            logger.error(f"Invalid date or time format for booking: {date}, {time}. Error: {e}")
            return f"Error: Invalid date or time format. Please use YYYY-MM-DD for date and HH:MM for time. Error: {str(e)}"
        except Error as e:
            logger.error(f"Database error booking appointment: {e}")
            return f"Error booking appointment: {str(e)}"
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()
    
    def get_appointments(self) -> str:
        """Get all appointments."""
        try:
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute('''
                SELECT name, DATE_FORMAT(date, '%Y-%m-%d') as date, TIME_FORMAT(time, '%H:%i') as time, purpose
                FROM appointments
                ORDER BY date, time
            ''') # Ensure date and time are formatted as strings
            appointments = cursor.fetchall()
            # cursor.close() # Closing in finally
            
            if not appointments:
                return "No appointments found."
            
            result = "Current Appointments:\n"
            for appt in appointments:
                result += f"- {appt['name']}: {appt['date']} at {appt['time']} ({appt['purpose']})\n"
            return result
        except Error as e:
            logger.error(f"Database error retrieving appointments: {e}")
            return f"Error retrieving appointments: {str(e)}"
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn.is_connected():
            self.conn.close()
            logger.info("Database connection closed.")
