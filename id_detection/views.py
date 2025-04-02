from django.shortcuts import render, redirect
from django.views import View
from django.http import JsonResponse, HttpResponse
from django.contrib.auth import login, authenticate
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.urls import reverse
import cv2
import numpy as np
from .utils import PersonDetector
from .models import Person, DetectionLog
from .forms import UserRegistrationForm, LoginForm
from PIL import Image
import io
from django.contrib import messages
from django.urls import reverse_lazy
from django.contrib.auth.views import LoginView as AuthLoginView
import base64
from django.views.generic import TemplateView

class RegisterView(View):
    def get(self, request):
        if request.user.is_authenticated:
            return redirect('detection')
        form = UserRegistrationForm()
        return render(request, 'id_detection/register.html', {'form': form})

    def post(self, request):
        form = UserRegistrationForm(request.POST, request.FILES)
        try:
            if form.is_valid():
                try:
                    # Create user
                    user = form.save()
                    
                    # Initialize detector
                    detector = PersonDetector()
                    
                    # Create person with verified images
                    person = Person.objects.create(
                        user=user,
                        designation=form.cleaned_data['designation'],
                        face_image=form.cleaned_data['face_image'],
                        id_card_image=form.cleaned_data['id_card_image']
                    )
                    
                    # Train face recognizer
                    if detector.train_face(person):
                        return JsonResponse({
                            'status': 'success',
                            'message': 'Registration successful! Please login with your credentials.',
                            'redirect_url': reverse('login')
                        })
                    else:
                        person.delete()
                        user.delete()
                        return JsonResponse({
                            'status': 'error',
                            'errors': 'Error training face recognition. Please try again with a clearer photo.'
                        })
                        
                except Exception as e:
                    if 'user' in locals():
                        user.delete()
                    return JsonResponse({
                        'status': 'error',
                        'errors': str(e)
                    })
            
            # If form is invalid, return form errors
            return JsonResponse({
                'status': 'error',
                'errors': form.errors
            })
        except Exception as e:
            # Catch any unexpected errors
            return JsonResponse({
                'status': 'error',
                'errors': f"An unexpected error occurred: {str(e)}"
            })

class LoginView(AuthLoginView):
    template_name = 'id_detection/login.html'
    form_class = LoginForm
    success_url = reverse_lazy('detection')

    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect('detection')
        return super().get(request, *args, **kwargs)

    def form_valid(self, form):
        # Get the user credentials
        username = form.cleaned_data.get('username')
        password = form.cleaned_data.get('password')
        
        # Authenticate user
        user = authenticate(username=username, password=password)
        
        if user is not None:
            login(self.request, user)
            messages.success(self.request, f'Welcome back, {username}!')
            return redirect('detection')
        else:
            messages.error(self.request, 'Invalid username or password')
            return self.form_invalid(form)

    def form_invalid(self, form):
        messages.error(self.request, 'Invalid username or password')
        return super().form_invalid(form)

@method_decorator(login_required, name='dispatch')
class DetectionView(View):
    def get(self, request):
        try:
            detection_logs = DetectionLog.objects.filter(
                person__user=request.user
            ).order_by('-timestamp')[:10]
        except:
            detection_logs = []
        return render(request, 'id_detection/monitor.html', {'detection_logs': detection_logs})

    def post(self, request):
        if 'image' not in request.FILES:
            return JsonResponse({
                'status': 'error',
                'message': 'No image provided'
            })

        try:
            # Make sure we're at the beginning of the file
            request.FILES['image'].seek(0)
            
            detector = PersonDetector()
            detections = detector.process_frame(request.FILES['image'])
            
            if detections:
                # Convert the processed frame to base64 for display
                _, buffer = cv2.imencode('.jpg', detections[0]['frame'])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return JsonResponse({
                    'status': 'success',
                    'detections': [
                        {
                            'person_name': d['person'].user.username,
                            'wearing_id': d['wearing_id'],
                            'timestamp': d['person'].last_detected.isoformat() if d['person'].last_detected else None
                        }
                        for d in detections
                    ],
                    'frame': f'data:image/jpeg;base64,{frame_base64}'
                })
            else:
                # Return empty frame if no detections
                request.FILES['image'].seek(0)  # Reset file position
                image_data = request.FILES['image'].read()
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    # If we can't decode the image, create a blank one
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)
                
                # Add "No detections" text
                cv2.putText(frame, "No persons detected", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                          
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return JsonResponse({
                    'status': 'success',
                    'detections': [],
                    'frame': f'data:image/jpeg;base64,{frame_base64}'
                })
        except Exception as e:
            print(f"Error in detection view: {str(e)}")
            # Create a blank image with error message
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Error: {str(e)}", (50, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return JsonResponse({
                'status': 'success',  # Return success to avoid breaking the frontend
                'detections': [],
                'frame': f'data:image/jpeg;base64,{frame_base64}',
                'error': str(e)
            }) 

class LandingPageView(TemplateView):
    template_name = 'id_detection/landing.html' 