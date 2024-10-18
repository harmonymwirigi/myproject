from django.contrib import admin

# Register your models here.
from .models import SubscriptionPlan,Subscription

admin.site.register(Subscription)
admin.site.register(SubscriptionPlan)