from django.contrib import admin
from django.urls import path, include
from user_auth import views as auth_views  # User authentication views
from subscription import views as subscription_views  # Subscription-related views
from .views import (
    home_page_view, about_us_view, optimization_us_view,
    case_study_us_view, explore_us_view, contact_us_view, profile_view
)

urlpatterns = [
    path('admin/', admin.site.urls),  # Admin path
    path("", home_page_view, name="home"),  # Home page view
    path("about/", about_us_view, name="about"),  # About us view
    path("login/", auth_views.login_view, name="login"),  # Login view
    path("activate/<uidb64>/<token>/", auth_views.activate, name="activate"),  # Login view
    path('logout/', auth_views.logout_view, name='logout'),  # Logout view
    path('profile/', profile_view, name='profile'),  # Profile view
    path("register/", auth_views.register_view, name="register"),  # Registration view
    path("payment/", subscription_views.payment_history, name="payment"),  # Payment history view
    path("payment_success/", auth_views.payment_success_view, name="payment_success"),  # Payment success view
    path("payment_cancel/", auth_views.payment_cancel_view, name="payment_cancel"),  # Payment cancel view
    path("webhook/", auth_views.stripe_webhook_view, name="webhook"),  # Stripe webhook view
    path("optimization/", optimization_us_view, name="optimization"),  # Optimization view
    path("casestudy/", case_study_us_view, name="casestudy"),  # Case study view
    path("explore/", explore_us_view, name="explore"),  # Explore view
    path("contact/", contact_us_view, name="contact"),  # Contact us view
    path('dashboard/', include('dashboard.urls')),  # Dashboard app URLs with namespace
    path('cancel-subscription/<int:subscription_id>/', subscription_views.cancel_subscription, name='cancel_subscription'),
    path('change-subscription/', subscription_views.change_subscription, name='change_subscription'),  # Change subscription view
    path('subscription_list/', subscription_views.subscription_list, name='subscription_list'),  # Change subscription view
    path('subscribe/<int:plan_id>/', subscription_views.subscribe, name='subscribe'),  # Change subscription view data_input_view
    
    
]
