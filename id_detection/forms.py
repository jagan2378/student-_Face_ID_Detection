from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from .models import Person
from .utils import PersonDetector

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    designation = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    face_image = forms.ImageField(
        help_text='Please look straight at the camera while taking photo'
    )
    id_card_image = forms.ImageField(
        help_text='Please provide a clear photo of your ID card'
    )
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add Bootstrap classes to the default fields
        self.fields['username'].widget.attrs.update({'class': 'form-control'})
        self.fields['password1'].widget.attrs.update({'class': 'form-control'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control'})

    def clean(self):
        cleaned_data = super().clean()
        face_image = cleaned_data.get('face_image')
        id_card_image = cleaned_data.get('id_card_image')

        if face_image and id_card_image:
            detector = PersonDetector()
            
            # Verify face image
            if not detector.verify_face_image(face_image):
                raise forms.ValidationError(
                    "We couldn't detect a clear face in your photo. Please ensure your face is centered, well-lit, and looking directly at the camera."
                )

            # Verify ID card image
            if not detector.verify_id_card_image(id_card_image):
                raise forms.ValidationError(
                    "We couldn't detect a valid ID card in your photo. Please ensure the entire ID card is visible, well-lit, and not blurry."
                )

        return cleaned_data

class LoginForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'Username'
    }))
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'form-control',
        'placeholder': 'Password'
    })) 