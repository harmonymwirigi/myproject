from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser
from subscription.models import SubscriptionPlan

class SignUpForm(UserCreationForm):
    phonenumber = forms.CharField(
        max_length=15,
        required=True,
        help_text="",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your phone number'
        })
    )
    businessname = forms.CharField(
        max_length=100,
        required=True,
        help_text="",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your business name'
        })
    )
    subscription_plan = forms.ModelChoiceField(
        queryset=SubscriptionPlan.objects.filter(active=True),  # Only show active plans
        required=True,
        empty_label="Select a Subscription Plan",
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )

    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'phonenumber', 'businessname', 'subscription_plan', 'password1', 'password2')
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your username'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your email'
            }),
        }
        help_texts = {
            'username': '',  # Remove help text for username
        }

    def __init__(self, *args, **kwargs):
        super(SignUpForm, self).__init__(*args, **kwargs)
        
        # Remove help texts for password fields
        self.fields['password1'].help_text = ''
        self.fields['password2'].help_text = ''
        
        # Apply consistent styling to password fields
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Enter your password'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Confirm your password'
        })


class LoginForm(forms.Form):
    username = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your username'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password'
        })
    )