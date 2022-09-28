from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path
from CoeuSearch import views
from django.views.static import serve

urlpatterns = [
    path('', views.home, name='home'),
    path('search', views.search, name='search')
]