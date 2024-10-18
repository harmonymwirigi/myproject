# dashboard/urls.py
from django.urls import path
from . import views  # Import your dashboard views
 # Namespace for the dashboard app
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),  # Dashboard home view
    path('optimize/', views.process_data, name='optimize'),  # Dashboard optimize view
    path('create-project/', views.create_project, name='create-project'),  # Dashboard optimize view select_project edit_project_data
    path('constraints/<int:project_id>/', views.constraints, name='constraints'),  
    path('project/<int:project_id>/', views.project_view, name='project'),  
    path('projects/', views.projects, name='projects'),  
    path('select_project/', views.select_project, name='select_project'),  
    path('create-set/<int:project_id>/', views.create_set, name='create-set'),  
    path('project/<int:project_id>/edit/', views.edit_project_data, name='edit_project_data'),
    path('uploadfile/<int:project_id>', views.uploadfile, name='uploadfile'),
    path('calender/<int:project_id>', views.calendar_view, name='calender'),
]
