# Generated by Django 5.1.5 on 2025-02-01 04:19

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Person',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('designation', models.CharField(max_length=100)),
                ('face_image', models.ImageField(upload_to='face_images/')),
                ('id_card_image', models.ImageField(upload_to='id_card_images/')),
                ('last_detected', models.DateTimeField(auto_now=True)),
                ('wearing_id', models.BooleanField(default=False)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='DetectionLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('wearing_id', models.BooleanField()),
                ('image', models.ImageField(upload_to='detection_logs/')),
                ('person', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='id_detection.person')),
            ],
        ),
    ]
