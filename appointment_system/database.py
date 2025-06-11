import mysql.connector
from mysql.connector import Error
from datetime import datetime

class AppointmentDB:
    def __init__(self, host='localhost', database='Appointments', user='root', password='root'):
        try:
            self.conn = mysql.connector.connect(
                host=host,
                database=database,
                user=user,
                password=password
            )
            self._setup_database()
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
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
            cursor.close()
            
            return f"Successfully booked appointment for {name} on {date} at {time}"
        except ValueError as e:
            return f"Error: Invalid date or time format. Please use YYYY-MM-DD for date and HH:MM for time. Error: {str(e)}"
        except Error as e:
            return f"Error booking appointment: {str(e)}"
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def get_appointments(self) -> str:
        """Get all appointments."""
        try:
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute('''
                SELECT name, date, time, purpose
                FROM appointments
                ORDER BY date, time
            ''')
            appointments = cursor.fetchall()
            cursor.close()
            
            if not appointments:
                return "No appointments found."
            
            # Format the appointments into a readable string
            result = "Current Appointments:\n"
            for appt in appointments:
                result += f"- {appt['name']}: {appt['date']} at {appt['time']} ({appt['purpose']})\n"
            return result
        except Error as e:
            return f"Error retrieving appointments: {str(e)}"
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn.is_connected():
            self.conn.close()
