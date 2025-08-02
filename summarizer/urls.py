from django.urls import path
from .views import summarize_and_answer


urlpatterns = [
    path('', summarize_and_answer, name='home'),
]