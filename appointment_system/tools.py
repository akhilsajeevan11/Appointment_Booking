from .database import AppointmentDB
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AppointmentTools:
    def __init__(self):
        self.db = AppointmentDB()
        
    def book_appointment(self, params):
        try:
            # Validate input data
            if not all(key in params for key in ['name', 'date', 'time', 'purpose']):
                return "Missing information: name, date, time, and purpose are required."
            
            # Book the appointment
            return self.db.book_appointment(
                params['name'],
                params['date'],
                params['time'],
                params['purpose']
            )
        except Exception as e:
            logger.error(f"Error booking appointment: {e}")
            return f"Error: {str(e)}"
        
    def view_appointments(self, _=None):
        try:
            appointments = self.db.get_appointments()
            if not appointments:
                return "No appointments found."
            
            # Format appointments for display
            formatted_appointments = []
            for appt in appointments:
                formatted_appointments.append(
                    f"Name: {appt['name']}\n"
                    f"Date: {appt['date']}\n"
                    f"Time: {appt['time']}\n"
                    f"Purpose: {appt['purpose']}\n"
                )
            return "\n".join(formatted_appointments)
        except Exception as e:
            logger.error(f"Error viewing appointments: {e}")
            return f"Error: {str(e)}"
        
    def handle_greeting(self, _=None):
        return """Welcome! I can help you:
1. Book a new appointment
2. View your existing appointments
3. Get examples of valid appointment purposes

What would you like to do?"""

    def get_purpose_examples(self, _=None):
        return """Here are some examples of valid appointment purposes:
1. Regular checkup
2. Annual physical examination
3. Dental cleaning
4. Eye examination
5. Vaccination
6. Blood test
7. X-ray
8. Consultation
9. Follow-up visit
10. Emergency care"""
    
    def get_tools(self):
        return [
            {
                "name": "BookAppointment",
                "func": self.book_appointment,
                "description": "Useful for booking appointments. Input should be a dictionary with 'name', 'date', 'time', and 'purpose' keys."
            },
            {
                "name": "ViewAppointments",
                "func": self.view_appointments,
                "description": "Useful for viewing all booked appointments. No input needed."
            },
            {
                "name": "HandleGreeting",
                "func": self.handle_greeting,
                "description": "Useful for handling greetings and welcome messages. No input needed."
            },
            {
                "name": "GetPurposeExamples",
                "func": self.get_purpose_examples,
                "description": "Useful for getting examples of valid appointment purposes. No input needed."
            }
        ]
