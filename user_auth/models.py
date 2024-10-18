from django.contrib.auth.models import AbstractUser, Group, Permission
from django.db import models

class CustomUser(AbstractUser):
    phonenumber = models.CharField(max_length=15, blank=True, null=True)
    businessname = models.CharField(max_length=100, blank=True, null=True)

    # Override groups and user_permissions with unique related_names
    groups = models.ManyToManyField(
        Group,
        related_name='customuser_set',  # Change the related_name here
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
        related_query_name='customuser',
    )
    user_permissions = models.ManyToManyField(
        Permission,
        related_name='customuser_set',  # Change the related_name here
        blank=True,
        help_text='Specific permissions for this user.',
        related_query_name='customuser',
    )
