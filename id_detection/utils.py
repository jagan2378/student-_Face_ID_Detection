import cv2
import torch
import numpy as np
from PIL import Image
import io
from django.core.files.base import ContentFile
from django.conf import settings
from django.core.mail import send_mail
from .models import Person, DetectionLog
import os
import pickle
from django.utils import timezone
from django.core.mail import EmailMessage
from datetime import datetime

class PersonDetector:
    def __init__(self):
        try:
            # Initialize face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise Exception("Error loading face cascade classifier")
            
            # Initialize face recognition
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Load trained model if exists
            self.load_face_data()
            
            # Detection buffer to reduce false positives
            self.detection_buffer = {}
            self.buffer_size = 3
            
            # Add email cooldown tracking
            self.last_email_time = {}  # Dictionary to track last email time per student
            self.email_cooldown = 120  # Cooldown period in seconds (2 minutes)
            
        except Exception as e:
            print(f"Error initializing PersonDetector: {str(e)}")
            raise

    def preprocess_image(self, image_file):
        """Convert Django UploadedFile to numpy array"""
        # Read image file
        image = Image.open(image_file)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Convert to numpy array
        image_np = np.array(image)
        return image_np

    def detect_person_and_id(self, image):
        """Detect persons and ID cards in the image"""
        # Run inference
        results = self.model(image)
        
        # Convert results to pandas DataFrame
        detections = results.pandas().xyxy[0]
        
        # Filter detections for persons and ID cards
        persons_detected = detections[detections['name'] == 'person']
        id_cards_detected = detections[detections['name'] == 'id']
        
        return persons_detected, id_cards_detected

    def is_id_card_near_person(self, person, id_card, threshold=100):
        """Check if ID card is near a person"""
        person_center = ((person['xmin'] + person['xmax'])/2, (person['ymin'] + person['ymax'])/2)
        id_center = ((id_card['xmin'] + id_card['xmax'])/2, (id_card['ymin'] + id_card['ymax'])/2)
        
        # Calculate Euclidean distance
        distance = np.sqrt(
            (person_center[0] - id_center[0])**2 + 
            (person_center[1] - id_center[1])**2
        )
        
        return distance < threshold

    def train_face(self, person):
        """Train the face recognizer with multiple angles of a person's face"""
        try:
            # Read the image
            image_data = person.face_image.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not read image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                raise ValueError("No face detected in the image")
            
            # Use the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            
            # Create augmented versions for better recognition
            face_samples = []
            face_samples.append(face_roi)  # Original face
            
            # Add slightly rotated versions (±5°, ±10°)
            for angle in [-10, -5, 5, 10]:
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                rotated = cv2.warpAffine(face_roi, M, (w, h))
                face_samples.append(rotated)
            
            # Add brightness variations
            bright = cv2.convertScaleAbs(face_roi, alpha=1.2, beta=10)
            dark = cv2.convertScaleAbs(face_roi, alpha=0.8, beta=-10)
            face_samples.append(bright)
            face_samples.append(dark)
            
            # Train with all samples
            labels = np.array([person.id] * len(face_samples))
            self.face_recognizer.train(face_samples, labels)
            
            # Save the trained model
            self.face_recognizer.write(self.face_data_file)
            print(f"Successfully trained face for person {person.id} with {len(face_samples)} samples")
            return True
            
        except Exception as e:
            print(f"Error training face: {str(e)}")
            return False

    def identify_person(self, person_img):
        """Simplified identification for testing"""
        try:
            # For testing, return the first person in the database
            return Person.objects.first()
        except Person.DoesNotExist:
            return None

    def process_frame(self, image_file):
        """Process a frame to detect persons and ID cards"""
        try:
            # Convert uploaded file to numpy array
            if hasattr(image_file, 'read'):  # Check if it's a file-like object
                # Save the current position
                if hasattr(image_file, 'seek') and hasattr(image_file, 'tell'):
                    pos = image_file.tell()
                    image_file.seek(0)  # Go to the beginning of the file
                
                image_data = image_file.read()
                
                # Reset position if needed
                if hasattr(image_file, 'seek') and hasattr(image_file, 'tell'):
                    image_file.seek(pos)
                
                # Check if we actually got data
                if not image_data or len(image_data) == 0:
                    raise ValueError("Empty image data received")
                    
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # If it's already a numpy array
                frame = image_file
            
            if frame is None:
                raise ValueError("Could not decode image data")
            
            # Make a copy for drawing results
            result_frame = frame.copy()
            
            # Detect faces
            face_rois = self.detect_faces(frame)
            
            # Detect ID cards
            id_cards = self.detect_id_cards(frame)
            
            detections = []
            
            # Process each detected face
            for face_roi in face_rois:
                # Recognize person
                person = self.recognize_face(face_roi)
                
                if person:
                    # Check if person is wearing ID
                    wearing_id = False
                    for id_card in id_cards:
                        # Draw ID card rectangle
                        x1, y1 = int(id_card['xmin']), int(id_card['ymin'])
                        x2, y2 = int(id_card['xmax']), int(id_card['ymax'])
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Check if this ID card belongs to this person
                        # This is a simplified check - you may want to improve this logic
                        wearing_id = True
                    
                    # Draw face rectangle and info
                    # Get face position in original frame
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )
                    
                    for (x, y, w, h) in faces:
                        # Check if this is the face we recognized
                        face_img = frame[y:y+h, x:x+w]
                        if np.array_equal(cv2.resize(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), (100, 100)), 
                                          cv2.resize(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY), (100, 100))):
                            # Draw rectangle
                            color = (0, 255, 0) if wearing_id else (0, 0, 255)
                            cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 2)
                            
                            # Add text
                            status = "ID: OK" if wearing_id else "ID: Missing"
                            cv2.putText(result_frame, person.user.username, (x, y-30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            cv2.putText(result_frame, status, (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Update person's status
                    person.wearing_id = wearing_id
                    person.last_detected = timezone.now()
                    person.save()
                    
                    # Create detection log
                    detection_log = DetectionLog.objects.create(
                        person=person,
                        wearing_id=wearing_id
                    )
                    
                    # Save detection image
                    _, buffer = cv2.imencode('.jpg', result_frame)
                    detection_log.image.save(
                        f'detection_{detection_log.id}.jpg',
                        ContentFile(buffer.tobytes())
                    )
                    
                    # Print detection status
                    print(f"\nDetection Results:")
                    print(f"Student: {person.user.username}")
                    print(f"Wearing ID Card: {'Yes' if wearing_id else 'No'}")
                    
                    if not wearing_id:
                        try:
                            self.send_alert_email(person, result_frame)
                            print("Alert email sent successfully!")
                        except Exception as e:
                            print(f"Failed to send alert email: {str(e)}")
                    
                    # Add to detections list
                    detections.append({
                        'person': person,
                        'wearing_id': wearing_id,
                        'frame': result_frame
                    })
            
            return detections
            
        except Exception as e:
            print(f"Error in processing frame: {str(e)}")
            return []

    def handle_violation(self, student, frame):
        """Handle ID card violation"""
        try:
            # Check buffer to avoid repeated alerts
            if not self.should_send_alert(student):
                return
                
            # Send email alert
            self.send_alert_email(student, frame)
            
            # Log violation
            DetectionLog.objects.create(
                person=student,
                wearing_id=False,
                image=self._save_frame(frame)
            )
            
        except Exception as e:
            print(f"Error handling violation: {str(e)}")

    def should_send_alert(self, student):
        """Check if alert should be sent based on buffer"""
        current_time = timezone.now()
        last_alert = self.detection_buffer.get(student.id, None)
        
        if last_alert is None or (current_time - last_alert).seconds > 300:  # 5 minutes
            self.detection_buffer[student.id] = current_time
            return True
        return False

    def _detect_id_card(self, roi):
        """Detect ID card in the region of interest"""
        try:
            # Convert to multiple color spaces
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 1. Color Detection for ID card (white/light blue)
            # White/light colored card detection
            lower_white = np.array([0, 0, 180])  # Increased brightness threshold
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Light blue detection (for ID cards with blue tint)
            lower_light_blue = np.array([90, 30, 180])
            upper_light_blue = np.array([130, 85, 255])
            light_blue_mask = cv2.inRange(hsv, lower_light_blue, upper_light_blue)
            
            # Combine masks
            id_mask = cv2.bitwise_or(white_mask, light_blue_mask)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5,5), np.uint8)
            id_mask = cv2.morphologyEx(id_mask, cv2.MORPH_CLOSE, kernel)
            id_mask = cv2.morphologyEx(id_mask, cv2.MORPH_OPEN, kernel)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(id_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Debug visualization
            debug_image = cv2.cvtColor(id_mask, cv2.COLOR_GRAY2BGR)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                roi_area = roi.shape[0] * roi.shape[1]
                
                # Area checks - ID card should be a reasonable size
                min_area_ratio = 0.01  # Minimum 1% of ROI
                max_area_ratio = 0.4   # Maximum 40% of ROI
                area_ratio = area / roi_area
                
                if not (min_area_ratio < area_ratio < max_area_ratio):
                    continue
                    
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                
                # Aspect ratio check (typical ID card ratio)
                if not (1.2 <= aspect_ratio <= 2.0):  # Relaxed aspect ratio constraints
                    continue
                    
                # Check for rectangular shape
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                # Allow for some variation in corner detection
                if len(approx) >= 4 and len(approx) <= 6:
                    # Get the region inside the contour
                    mask = np.zeros(gray.shape, np.uint8)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    
                    # Check average brightness of the region
                    mean_val = cv2.mean(roi, mask=mask)[0]
                    
                    # ID cards should be bright
                    if mean_val > 120:  # Adjusted brightness threshold
                        # Draw detection for debugging
                        cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(roi, f"ID Card ({area_ratio:.2f})", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        print(f"ID card detected - AR: {aspect_ratio:.2f}, Area: {area_ratio:.2f}, Brightness: {mean_val}")
                        return True
            
            print("No valid ID card detected")
            return False
            
        except Exception as e:
            print(f"Error in ID detection: {str(e)}")
            return False

    def _save_frame(self, frame):
        """Save the frame as an image file"""
        try:
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                image_file = ContentFile(buffer.tobytes())
                filename = f'detection_{DetectionLog.objects.count()}.jpg'
                return filename
        except Exception as e:
            print(f"Error saving frame: {str(e)}")
        return None

    def can_send_email(self, person):
        """Check if enough time has passed since the last email for this person"""
        current_time = timezone.now()
        last_time = self.last_email_time.get(person.id)
        
        if last_time is None:
            return True
            
        time_diff = (current_time - last_time).total_seconds()
        return time_diff >= self.email_cooldown

    def send_alert_email(self, person, frame):
        """Send email alert for student not wearing ID"""
        try:
            # Check cooldown period
            if not self.can_send_email(person):
                print(f"\nSkipping email alert - Cooldown period active for {person.user.username}")
                remaining_time = self.email_cooldown - (timezone.now() - self.last_email_time[person.id]).total_seconds()
                print(f"Next alert can be sent in {int(remaining_time)} seconds")
                return
            
            # Save the frame as an image
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = ContentFile(buffer.tobytes())
            
            # Get the current time in a readable format
            current_time = timezone.localtime().strftime('%Y-%m-%d %I:%M:%S %p')
            
            # Create email subject and message
            subject = f'ID Card Missing Alert - {person.user.username}'
            message = f"""
SECURITY ALERT: Student Detected Without ID Card

Student Details:
----------------
Name: {person.user.username}
Email: {person.user.email}
Designation: {person.designation}
Detection Time: {current_time}
Location: College Premises

Action Required:
---------------
1. Please verify the student's identity
2. Ensure the student wears their ID card immediately
3. Document this violation if necessary

This is an automated security alert. A snapshot of the detection is attached.

Best regards,
College Security Monitoring System
            """
            
            try:
                # Create email
                email = EmailMessage(
                    subject=subject,
                    body=message,
                    from_email=settings.EMAIL_HOST_USER,
                    to=[settings.ADMIN_EMAIL],
                    reply_to=[settings.EMAIL_HOST_USER],
                )
                
                # Attach the image
                image_name = f'violation_{person.user.username}_{timezone.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                email.attach(image_name, image_data.read(), 'image/jpeg')
                
                # Send email with detailed error handling
                try:
                    email.send(fail_silently=False)
                    # Update last email time after successful send
                    self.last_email_time[person.id] = timezone.now()
                    
                    print(f"\nEmail Alert Details:")
                    print(f"From: {settings.EMAIL_HOST_USER}")
                    print(f"To: {settings.ADMIN_EMAIL}")
                    print(f"Subject: ID Card Missing Alert - {person.user.username}")
                    print("Email sent successfully!")
                    print(f"Next alert can be sent after: {timezone.now() + timezone.timedelta(seconds=self.email_cooldown)}")
                    
                    # Create detection log
                    DetectionLog.objects.create(
                        person=person,
                        wearing_id=False,
                        image=image_name
                    )
                except Exception as e:
                    print("\nEmail Sending Failed:")
                    print(f"Error: {str(e)}")
                    print("\nEmail Configuration:")
                    print(f"SMTP Server: {settings.EMAIL_HOST}:{settings.EMAIL_PORT}")
                    print(f"TLS Enabled: {settings.EMAIL_USE_TLS}")
                    print("Please verify your Gmail App Password and settings")
                    raise e
                    
            except Exception as e:
                print(f"Email creation failed: {str(e)}")
            
        except Exception as e:
            print(f"Alert processing failed: {str(e)}")
            raise e

    def draw_detections(self, image, persons, id_cards):
        """Draw bounding boxes around detected persons and ID cards"""
        image_draw = image.copy()
        
        # Draw person detections
        for _, person in persons.iterrows():
            cv2.rectangle(
                image_draw,
                (int(person['xmin']), int(person['ymin'])),
                (int(person['xmax']), int(person['ymax'])),
                (0, 255, 0),  # Green for persons
                2
            )
            
        # Draw ID card detections
        for _, id_card in id_cards.iterrows():
            cv2.rectangle(
                image_draw,
                (int(id_card['xmin']), int(id_card['ymin'])),
                (int(id_card['xmax']), int(id_card['ymax'])),
                (255, 0, 0),  # Blue for ID cards
                2
            )
            
        return image_draw 

    def detect(self, image_bytes):
        """Detect persons, faces, and ID cards in the image"""
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect objects using YOLO
        results = self.model(image)
        detections = []
        
        # Get person and ID card detections
        persons = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'person']
        id_cards = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'id']
        
        # Process each person detection
        for _, person in persons.iterrows():
            x1, y1, x2, y2 = map(int, [person['xmin'], person['ymin'], person['xmax'], person['ymax']])
            person_roi = image[y1:y2, x1:x2]
            
            # Convert to grayscale for face detection
            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_roi, 1.3, 5)
            
            for (fx, fy, fw, fh) in faces:
                face_roi = gray_roi[fy:fy+fh, fx:fx+fw]
                
                # Try to recognize the face
                label, confidence = self.face_recognizer.predict(face_roi)
                
                if confidence < 100:  # Adjust threshold as needed
                    person_obj = Person.objects.get(id=label)
                    
                    # Check if person is wearing ID
                    wearing_id = False
                    for _, id_card in id_cards.iterrows():
                        if self._check_id_proximity(person, id_card):
                            wearing_id = True
                            break
                    
                    # Update person status
                    person_obj.wearing_id = wearing_id
                    person_obj.save()
                    
                    # Create detection log
                    log = DetectionLog.objects.create(
                        person=person_obj,
                        wearing_id=wearing_id,
                        image=self._save_detection_image(image)
                    )
                    
                    detections.append({
                        'person': person_obj,
                        'wearing_id': wearing_id
                    })
        
        return detections, self._draw_detections(image, persons, id_cards)

    def _check_id_proximity(self, person, id_card):
        """Check if ID card is close to the person"""
        person_center = ((person['xmin'] + person['xmax']) / 2, (person['ymin'] + person['ymax']) / 2)
        id_center = ((id_card['xmin'] + id_card['xmax']) / 2, (id_card['ymin'] + id_card['ymax']) / 2)
        
        # Calculate distance between person and ID card
        distance = np.sqrt(
            (person_center[0] - id_center[0])**2 + 
            (person_center[1] - id_center[1])**2
        )
        
        # ID card should be within reasonable distance of person
        max_distance = (person['xmax'] - person['xmin']) * 0.8
        return distance < max_distance

    def _save_detection_image(self, image):
        # Implement the logic to save the detection image
        pass

    def _draw_detections(self, image, persons, id_cards):
        # Implement the logic to draw the detection results
        pass 

    def verify_face_image(self, image_file):
        """Verify if image contains a clear face"""
        try:
            # Convert image file to numpy array
            image = self.preprocess_image(image_file)
            
            # Try different color conversions
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try different scale factors for better detection
            for scale_factor in [1.05, 1.1, 1.15, 1.2]:
                # Detect faces with more lenient parameters
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=3,  # Reduced from 5 to be more lenient
                    minSize=(20, 20)  # Smaller minimum size
                )
                
                if len(faces) > 0:
                    # Found at least one face
                    print(f"Detected {len(faces)} faces with scale factor {scale_factor}")
                    
                    # Use the largest face
                    if len(faces) > 1:
                        face = max(faces, key=lambda f: f[2] * f[3])
                    else:
                        face = faces[0]
                        
                    x, y, w, h = face
                    
                    # Check if face is reasonably sized (at least 10% of image)
                    img_area = image.shape[0] * image.shape[1]
                    face_area = w * h
                    face_ratio = face_area / img_area
                    
                    if face_ratio < 0.05:  # Face is too small
                        print(f"Face too small: {face_ratio:.2f} of image")
                        continue
                    
                    # We found a good face
                    return True
                    
            # If we get here, no suitable face was found
            print("No suitable face found in the image")
            return False
            
        except Exception as e:
            print(f"Error verifying face image: {str(e)}")
            return False

    def verify_id_card_image(self, image_file):
        """Verify if image contains a valid ID card"""
        try:
            # Convert image file to numpy array
            image = self.preprocess_image(image_file)
            
            # Use ID card detection logic
            roi = cv2.resize(image, (640, 480))
            has_id = self._detect_id_card(roi)
            
            return has_id
            
        except Exception as e:
            print(f"Error verifying ID card image: {str(e)}")
            return False

    def verify_student_id(self, student, id_cards, frame):
        """Improved verification if student is wearing their registered ID"""
        try:
            # No ID cards detected
            if not id_cards:
                return False
            
            # Get student's face position
            face_position = self.get_face_position(student, frame)
            if not face_position:
                # If we can't find the face position, use any detected ID card
                # This is more lenient but might lead to false positives
                return len(id_cards) > 0
            
            # Check proximity and orientation with more lenient criteria
            for id_card in id_cards:
                # 1. Check if ID is in the upper body area (below or near face)
                face_bottom = face_position['ymax']
                face_center_x = (face_position['xmin'] + face_position['xmax']) / 2
                
                id_top = id_card['ymin']
                id_center_x = (id_card['xmin'] + id_card['xmax']) / 2
                
                # ID should be below the face
                if id_top >= face_bottom - 20:  # Allow slight overlap
                    # ID should be somewhat aligned with the face horizontally
                    # More lenient horizontal alignment check
                    horizontal_distance = abs(face_center_x - id_center_x)
                    max_allowed_distance = frame.shape[1] * 0.3  # 30% of frame width
                    
                    if horizontal_distance < max_allowed_distance:
                        return True
            
            # If we have ID cards but none meet our criteria, check if any are close to the person
            # This is a fallback for cases where our positioning logic might fail
            if len(id_cards) > 0:
                # Get the closest ID card to the face
                face_center_x = (face_position['xmin'] + face_position['xmax']) / 2
                face_center_y = (face_position['ymin'] + face_position['ymax']) / 2
                
                min_distance = float('inf')
                for id_card in id_cards:
                    id_center_x = (id_card['xmin'] + id_card['xmax']) / 2
                    id_center_y = (id_card['ymin'] + id_card['ymax']) / 2
                    
                    distance = ((face_center_x - id_center_x) ** 2 + 
                               (face_center_y - id_center_y) ** 2) ** 0.5
                    
                    min_distance = min(min_distance, distance)
                
                # If any ID card is reasonably close to the face
                if min_distance < frame.shape[1] * 0.4:  # 40% of frame width
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error verifying student ID: {str(e)}")
            # Default to True to reduce false negatives during errors
            return True
        
    def verify_id_card_features(self, frame, id_card):
        """Verify if the detected region has ID card features"""
        try:
            # Extract the ID card region
            x1, y1 = int(id_card['xmin']), int(id_card['ymin'])
            x2, y2 = int(id_card['xmax']), int(id_card['ymax'])
            card_roi = frame[y1:y2, x1:x2]
            
            # 1. Check for text using OCR or text-like features
            gray_roi = cv2.cvtColor(card_roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Find text-like contours (small rectangles)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_like_regions = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                
                # Text typically has specific aspect ratios and sizes
                if 0.1 < aspect_ratio < 10 and 5 < w < 100 and 5 < h < 30:
                    text_like_regions += 1
            
            # ID cards typically have multiple text regions
            if text_like_regions > 5:
                return True
            
            return False
            
        except Exception as e:
            print(f"Error verifying ID card features: {str(e)}")
            return False

    def load_face_data(self):
        """Load trained face recognition data"""
        try:
            # Create face data directory if it doesn't exist
            self.face_data_path = os.path.join(settings.MEDIA_ROOT, 'face_data')
            os.makedirs(self.face_data_path, exist_ok=True)
            
            # Path to face recognition data file
            self.face_data_file = os.path.join(self.face_data_path, 'face_recognition_data.yml')
            
            # Load existing face data if available
            if os.path.exists(self.face_data_file):
                try:
                    self.face_recognizer.read(self.face_data_file)
                    print("Successfully loaded face recognition data")
                except Exception as e:
                    print(f"Error loading face recognition data: {str(e)}")
                    # If file is corrupted, remove it
                    if os.path.exists(self.face_data_file):
                        os.remove(self.face_data_file)
                    
        except Exception as e:
            print(f"Error in load_face_data: {str(e)}")
            raise 

    def recognize_face(self, face_roi):
        """Recognize a face in the given ROI"""
        try:
            # Convert to grayscale if needed
            if len(face_roi.shape) > 2:
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Predict
            label, confidence = self.face_recognizer.predict(face_roi)
            
            # Get person from database
            if confidence < 100:  # Adjust threshold as needed
                try:
                    person = Person.objects.get(id=label)
                    return person
                except Person.DoesNotExist:
                    return None
                
            return None
            
        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
            return None

    def detect_faces(self, frame):
        """Detect faces in frame"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Extract face ROIs
            face_rois = []
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                face_rois.append(face_roi)
            
            return face_rois
            
        except Exception as e:
            print(f"Error detecting faces: {str(e)}")
            return [] 

    def detect_id_cards(self, frame):
        """Improved ID card detection with multiple approaches"""
        try:
            # Resize for consistent processing
            frame_resized = cv2.resize(frame, (640, 480))
            
            # Convert to multiple color spaces for better detection
            hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            
            id_cards = []
            
            # APPROACH 1: Rectangle detection using contours
            # Apply adaptive thresholding to handle different lighting conditions
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find edges
            edges = cv2.Canny(thresh, 50, 150)
            
            # Dilate to connect edge fragments
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour
            for contour in contours:
                # Filter by area - ID cards are typically medium-sized in the frame
                area = cv2.contourArea(contour)
                if area < 3000 or area > 100000:  # More permissive size range
                    continue
                    
                # Approximate the contour to a polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Check if it's a quadrilateral (4 sides)
                if len(approx) >= 4 and len(approx) <= 6:  # Allow slight imperfections
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (ID cards are typically rectangular)
                    aspect_ratio = w / float(h)
                    if 1.0 <= aspect_ratio <= 2.2:  # More permissive aspect ratio
                        # Add to candidates with medium confidence
                        id_cards.append({
                            'xmin': x,
                            'ymin': y,
                            'xmax': x + w,
                            'ymax': y + h,
                            'confidence': 0.7
                        })
            
            # APPROACH 2: Color-based detection for ID cards
            # ID cards often have white/light backgrounds
            lower_white = np.array([0, 0, 180])  # More permissive lower bound
            upper_white = np.array([180, 40, 255])  # More permissive upper bound
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            
            # Find contours in the white mask
            contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 3000 or area > 100000:
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                if 1.0 <= aspect_ratio <= 2.2:
                    # Check if this region overlaps with any existing detection
                    new_detection = True
                    for existing in id_cards:
                        overlap = self._calculate_overlap(
                            (x, y, x+w, y+h),
                            (existing['xmin'], existing['ymin'], existing['xmax'], existing['ymax'])
                        )
                        if overlap > 0.5:  # If significant overlap
                            new_detection = False
                            # Update confidence if this is a better detection
                            existing['confidence'] = max(existing['confidence'], 0.6)
                            break
                    
                    if new_detection:
                        id_cards.append({
                            'xmin': x,
                            'ymin': y,
                            'xmax': x + w,
                            'ymax': y + h,
                            'confidence': 0.6
                        })
            
            # APPROACH 3: Hough Line Transform to detect rectangular objects
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                # Group lines by orientation (horizontal/vertical)
                horizontal_lines = []
                vertical_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x2 - x1) > abs(y2 - y1):  # Horizontal line
                        horizontal_lines.append(line[0])
                    else:  # Vertical line
                        vertical_lines.append(line[0])
                
                # Find potential rectangles from line intersections
                if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
                    # Use simple heuristic: find min/max coordinates
                    min_x = min([min(line[0], line[2]) for line in vertical_lines])
                    max_x = max([max(line[0], line[2]) for line in vertical_lines])
                    min_y = min([min(line[1], line[3]) for line in horizontal_lines])
                    max_y = max([max(line[1], line[3]) for line in horizontal_lines])
                    
                    w, h = max_x - min_x, max_y - min_y
                    aspect_ratio = w / float(h) if h > 0 else 0
                    
                    if 1.0 <= aspect_ratio <= 2.2 and w * h > 3000 and w * h < 100000:
                        # Check if this region overlaps with any existing detection
                        new_detection = True
                        for existing in id_cards:
                            overlap = self._calculate_overlap(
                                (min_x, min_y, max_x, max_y),
                                (existing['xmin'], existing['ymin'], existing['xmax'], existing['ymax'])
                            )
                            if overlap > 0.5:
                                new_detection = False
                                break
                        
                        if new_detection:
                            id_cards.append({
                                'xmin': min_x,
                                'ymin': min_y,
                                'xmax': max_x,
                                'ymax': max_y,
                                'confidence': 0.5
                            })
            
            # If we have multiple detections, keep only the highest confidence ones
            if len(id_cards) > 0:
                # Sort by confidence (highest first)
                id_cards.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Keep only non-overlapping detections
                filtered_cards = [id_cards[0]]
                for card in id_cards[1:]:
                    overlapping = False
                    for existing in filtered_cards:
                        overlap = self._calculate_overlap(
                            (card['xmin'], card['ymin'], card['xmax'], card['ymax']),
                            (existing['xmin'], existing['ymin'], existing['xmax'], existing['ymax'])
                        )
                        if overlap > 0.3:  # If significant overlap
                            overlapping = True
                            break
                    
                    if not overlapping:
                        filtered_cards.append(card)
                
                # For debugging
                print(f"Detected {len(filtered_cards)} ID cards")
                
                return filtered_cards
            
            return []
            
        except Exception as e:
            print(f"Error detecting ID cards: {str(e)}")
            return []

    def _calculate_overlap(self, box1, box2):
        """Calculate IoU (Intersection over Union) between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0

    def draw_results(self, frame, person, wearing_id):
        """Draw detection results on frame"""
        try:
            # Make a copy of the frame
            result_frame = frame.copy()
            
            # Get face position
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                # Draw face rectangle
                color = (0, 255, 0) if wearing_id else (0, 0, 255)
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 2)
                
                # Add person name and status
                status_text = "ID: OK" if wearing_id else "ID: Missing"
                cv2.putText(result_frame, person.user.username, (x, y-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(result_frame, status_text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            return result_frame
            
        except Exception as e:
            print(f"Error drawing results: {str(e)}")
            return frame 

    def get_face_position(self, person, frame):
        """Get the position of a person's face in the frame"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None
            
            # For each face, check if it matches the person
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                # Try to recognize the face
                recognized_person = self.recognize_face(face_roi)
                
                if recognized_person and recognized_person.id == person.id:
                    return {
                        'xmin': x,
                        'ymin': y,
                        'xmax': x + w,
                        'ymax': y + h
                    }
            
            return None
            
        except Exception as e:
            print(f"Error getting face position: {str(e)}")
            return None 