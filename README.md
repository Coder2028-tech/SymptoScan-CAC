SymptoScan
A web-based health symptom tracking app that uses device camera scanning to help users monitor facial symptoms over time.
What it does
Users scan facial regions — under the eyes, cheeks, and lips — using their device camera. The app tracks changes in these regions over time, helping users identify gradual symptom progression that might otherwise go unnoticed.
Why we built it
Subtle health changes — like those from a concussion, illness, or other conditions — often go undetected because no one is systematically tracking them. SymptoScan was built to close that gap, making symptom monitoring accessible on any device.
Tech Stack

Python — backend logic
Flask — web framework
SQLAlchemy — database ORM
SQLite — local data storage
HTML/CSS/JavaScript — frontend interface
OpenCV — camera integration and facial region detection

Key Features

Device camera integration for facial scanning
Detection of facial regions including under eyes, cheeks, and lips
Persistent symptom storage using SQLite database
Historical tracking of symptom changes over time
Cross-device compatible interface

How it works

User opens the app on any device with a camera
App accesses camera and scans designated facial regions
Detection parameters account for hardware variability across devices
Symptom data is stored persistently in SQLite database via SQLAlchemy
User can track changes over time through historical records

Files

app.py — Flask application and routing
models.py — SQLAlchemy database models
detection.py — camera integration and facial region detection
templates/ — frontend HTML interface
static/ — CSS and JavaScript files

How to run

bashpip install flask sqlalchemy opencv-python
python main.py
Then open http://localhost:5000 in your browser.

Background

Built by a three-person Girls Who Code team over sixteen weeks during summer and fall 2025 for the Congressional App Challenge. Awarded Honorable Mention in the 2025 Congressional App Challenge.
Technical challenges
Camera sensitivity calibration — different devices have varying camera sensitivity thresholds. We adjusted detection parameters to account for hardware variability across devices, learning that software assumptions about hardware consistency don't always hold in the real world.
Database integration — implemented persistent data storage using SQLAlchemy connected to SQLite, enabling symptom tracking across multiple sessions.

Team

Built collaboratively by a three-person Girls Who Code team at Ridge High School, NJ.
