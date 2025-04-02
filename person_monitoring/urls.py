from django.contrib import admin
from django.urls import path, include
from django.contrib.auth.views import LogoutView
from id_detection.views import RegisterView, LoginView, DetectionView, LandingPageView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', LandingPageView.as_view(), name='landing'),
    path('monitor/', DetectionView.as_view(), name='detection'),
    path('detect/', DetectionView.as_view(), name='detect'),
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(next_page='landing'), name='logout'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) 