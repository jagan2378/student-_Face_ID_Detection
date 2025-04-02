# Student ID Monitoring System

## Overview
The Student ID Monitoring System is a Django-based application that uses computer vision to detect and verify whether students are wearing their ID cards. The system captures video from a webcam, identifies students through facial recognition, and checks if they are wearing their ID cards. If a student is detected without an ID card, the system logs the violation and sends an email alert to administrators.

![System Screenshot](screenshots/dashboard.png)

## Features
- **User Authentication**: Secure registration and login system
- **Face Detection & Recognition**: Identifies registered students using facial recognition
- **ID Card Detection**: Detects ID cards using computer vision techniques
- **Real-time Monitoring**: Live video feed with real-time detection results
- **Violation Alerts**: Automated email notifications when students are detected without ID cards
- **Detection Logging**: Records all detection events with timestamps and images
- **Admin Dashboard**: Comprehensive admin interface for system management
- **Responsive Design**: Works on desktop and mobile devices

## Technology Stack
- **Backend**: Django (Python web framework)
- **Computer Vision**: OpenCV, YOLO object detection
- **Face Recognition**: OpenCV's LBPH Face Recognizer
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Database**: SQLite (default), compatible with PostgreSQL
- **Email**: SMTP integration for sending alerts

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or camera device
- Git (optional)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/student-id-monitoring.git
   cd student-id-monitoring
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install django
   pip install numpy
   pip install pillow
   pip install opencv-contrib-python  # Important: use contrib version for face recognition
   pip install torch torchvision  # For YOLO model
   ```

4. **Create necessary directories**
   ```bash
   mkdir -p media/face_data
   mkdir -p media/face_images
   mkdir -p media/id_card_images
   mkdir -p media/detection_logs
   mkdir -p static/css
   ```

5. **Apply database migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Create superuser (admin)**
   ```bash
   python manage.py createsuperuser
   ```

7. **Configure email settings**
   
   Edit `person_monitoring/settings.py` and update the email configuration:
   ```python
   EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
   EMAIL_HOST = 'smtp.gmail.com'
   EMAIL_PORT = 587
   EMAIL_USE_TLS = True

   .env file:
   
   EMAIL_HOST_USER = 'your-email@gmail.com'  # Change to your email
   EMAIL_HOST_PASSWORD = 'your-app-password'  # Use app password for Gmail
   ADMIN_EMAIL = 'admin-email@example.com'  # Admin email to receive alerts
   ```

8. **YOLO Model Setup**
   ```bash
   # Download YOLOv5 weights (if not included)
   mkdir -p yolo_model
   wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -O yolo_model/yolov5s.pt

   # Or use custom trained model if available
   # Place your custom .pt file in the yolo_model directory
   ```

9. **Collect static files**
   ```bash
   python manage.py collectstatic
   ```

10. **Run the development server**
    ```bash
    python manage.py runserver
    ```

11. **Access the application**
    
    Open your browser and navigate to: http://127.0.0.1:8000/

## Usage

### Registration
1. Navigate to the registration page
2. Fill in your details (username, email, password)
3. Upload a clear face photo (look directly at the camera)
4. Upload a clear photo of your ID card
5. Submit the form

### Login
1. Enter your username and password
2. Click "Login"

### Monitoring
1. After login, you'll be directed to the monitoring page
2. Click "Start Scan" to begin monitoring
3. The system will detect faces and ID cards in real-time
4. Status updates will appear on the right side of the screen
5. Click "Stop Scan" to end the monitoring session

### Admin Interface
1. Navigate to http://127.0.0.1:8000/admin/
2. Login with superuser credentials
3. Manage users, persons, and detection logs

## System Architecture

### Models
- **User**: Django's built-in user model for authentication
- **Person**: Extends User with additional fields like designation, face image, and ID card image
- **DetectionLog**: Records detection events with timestamps and violation images

### Components
- **Face Detection**: Uses Haar Cascade classifiers to detect faces in video frames
- **Face Recognition**: LBPH Face Recognizer identifies registered students
- **ID Card Detection**: Uses computer vision techniques to detect ID cards
- **Alert System**: Sends email notifications for violations
- **Logging System**: Records all detection events in the database

## Troubleshooting

### Camera Access Issues
- Ensure your browser has permission to access the camera
- Try a different browser if camera access fails
- Check if another application is using the camera

### Face Recognition Issues
- Ensure good lighting for clear face detection
- Look directly at the camera during registration
- If recognition fails, try re-registering with a clearer photo

### ID Card Detection Issues
- Make sure ID card is clearly visible
- Hold ID card near your face for better detection
- Ensure good lighting and contrast

### Email Alert Issues
- Verify email settings in settings.py
- For Gmail, ensure you're using an App Password
- Check spam folder for alert emails

### Database Issues
- If you encounter database errors, try:
  ```bash
  python manage.py makemigrations id_detection
  python manage.py migrate
  ```

### Performance Optimization
If the system runs slowly:

## Deployment

For production deployment:
1. Set `DEBUG = False` in settings.py
2. Configure a proper database (PostgreSQL recommended)
3. Set up proper email backend
4. Use a production-ready web server (Gunicorn, uWSGI)
5. Set up static file serving with Nginx
6. Configure proper security settings (HTTPS, etc.)

## Security Notes
- Regularly update dependencies
- Use strong passwords
- Consider implementing rate limiting
- Implement proper access controls
- Encrypt sensitive data

## Browser Compatibility
- Chrome and Firefox work best for camera access
- Safari may require additional permissions
- Ensure JavaScript is enabled in your browser

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- OpenCV for computer vision capabilities
- Django for the web framework
- Bootstrap for the responsive UI
- YOLOv5 for object detection

## Contact
For questions or support, please contact:
- Email: your-email@example.com
- GitHub: [Your GitHub Profile](https://github.com/yourusername)

---

© 2023 Student ID Monitoring System. All rights reserved.

## Accuracy Improvements

The system includes several enhancements to improve detection accuracy:

- **Data Augmentation**: Training with multiple variations of face images
- **Multi-technique ID Detection**: Using color, shape, and feature analysis
- **Spatial Relationship Verification**: Checking proper positioning of ID cards
- **Feature Verification**: Confirming ID card-specific features
- **Confidence Thresholds**: Filtering low-confidence detections
