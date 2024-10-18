from django import forms
from .models import Project, Set, Skill, Duration

class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ['name', 'start_date', 'finish_date']

class SetForm(forms.ModelForm):
    class Meta:
        model = Set
        fields = ['projects', 'teamleads', 'timeperiod', 'skills', 'managers']

class SkillForm(forms.ModelForm):
    class Meta:
        model = Skill
        fields = ['skills', 'teamleads', 'value']

class DurationForm(forms.ModelForm):
    class Meta:
        model = Duration
        fields = ['projects', 'value']
