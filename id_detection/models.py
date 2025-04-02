from django.db import models
from django.contrib.auth.models import User

class Person(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    designation = models.CharField(max_length=100)
    face_image = models.ImageField(upload_to='face_images/')
    id_card_image = models.ImageField(upload_to='id_card_images/')
    last_detected = models.DateTimeField(auto_now=True)
    wearing_id = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} - {self.designation}"

class DetectionLog(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    wearing_id = models.BooleanField()
    image = models.ImageField(upload_to='detection_logs/')

    def __str__(self):
        return f"{self.person.user.username} - {self.timestamp}" 