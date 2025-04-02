from django.contrib import admin
from .models import Person, DetectionLog

@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ('user', 'designation', 'wearing_id', 'last_detected')
    search_fields = ('user__username', 'designation')
    list_filter = ('wearing_id',)

@admin.register(DetectionLog)
class DetectionLogAdmin(admin.ModelAdmin):
    list_display = ('person', 'timestamp', 'wearing_id')
    list_filter = ('wearing_id', 'timestamp')
    search_fields = ('person__user__username',) 