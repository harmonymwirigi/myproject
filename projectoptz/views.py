from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.contrib import messages
from user_auth.models import CustomUser
from subscription.models import Subscription

User = get_user_model()

def home_page_view(request, *args, **kwargs):
    html_template = "home.html"
    return render(request, html_template)

def about_us_view(request):

    return render(request, 'about.html')

def optimization_us_view(request):
    
    return render(request, 'optimization.html')

def case_study_us_view(request):
    
    return render(request, 'case_study.html')

def explore_us_view(request):
    
    return render(request, 'explore.html')

def contact_us_view(request):
    
    return render(request, 'contact.html')

@login_required
def profile_view(request):
    user = request.user  # Get the currently logged-in user

    # Get the user's active subscription
    try:
        subscription = Subscription.objects.filter(user=user, status='active').first()
    except Subscription.DoesNotExist:
        subscription = None

    return render(request, 'dashboard/profile.html', {
        'user': user,
        'subscription': subscription
    })

