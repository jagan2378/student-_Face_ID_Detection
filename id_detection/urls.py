from django.urls import path
from django.contrib.auth.views import LogoutView
from .views import DetectionView, RegisterView, LoginView, LandingPageView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', LandingPageView.as_view(), name='landing'),
    path('monitor/', DetectionView.as_view(), name='detection'),
    path('detect/', DetectionView.as_view(), name='detect'),
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(next_page='landing'), name='logout'),
]

# Add this to serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 