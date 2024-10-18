from django.db import models
from django.utils import timezone
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from datetime import timedelta

class Project(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='projects', null=True)
    name = models.CharField(max_length=100, null=True)
    start_date = models.DateField(default=timezone.now, null=True)
    finish_date = models.DateField(default=timezone.now, null=True)

    def __str__(self):
        return self.name

class ProjectFiles(models.Model):
    FILE_TYPE_CHOICES = [
        ('plot', 'Plot File'),
        ('excel', 'Excel File')
    ]

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='files', null=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='projectsfiles', null=True)
    filename = models.CharField(max_length=200, null=True)
    file = models.FileField(null=True, blank=True)
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES, null=True, blank=True)

    def __str__(self):
        return self.filename


class Set(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='sets', null=True)
    projects = models.IntegerField(null=True)
    teamleads = models.IntegerField(null=True)
    timeperiod = models.IntegerField(null=True)
    skills = models.IntegerField(null=True)
    managers = models.IntegerField(null=True)

class Skill(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='skills', null=True)
    skills = models.CharField(max_length=100, null=True)
    teamleads = models.CharField(max_length=100, null=True)
    value = models.IntegerField(null=True)

class Duration(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='durations', null=True)
    projects = models.CharField(max_length=100, null=True)
    value = models.IntegerField(null=True)

class Reqskill(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='reqskills', null=True)
    skills = models.CharField(max_length=100, null=True)
    projects = models.CharField(max_length=100, null=True)
    value = models.IntegerField(null=True)

class MGskill(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='mgskills', null=True)
    skills = models.CharField(max_length=100, null=True)
    managers = models.CharField(max_length=100, null=True)
    value = models.IntegerField(null=True)

class DS(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='ds', null=True)
    projects = models.CharField(max_length=100, null=True)
    value = models.IntegerField(null=True)

class FS(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='fs', null=True)
    projects = models.CharField(max_length=100, null=True)
    value = models.IntegerField(null=True)

class WAD(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='wad', null=True)
    teamleads = models.CharField(max_length=100, null=True)
    value = models.IntegerField(null=True)

class Score(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='score', null=True)
    teamleads = models.CharField(max_length=100, null=True)
    value = models.FloatField(null=True)

class Cost(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='cost', null=True)
    teamleads = models.CharField(max_length=100, null=True)
    value = models.IntegerField(null=True)

class PreferenceCost(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='preference_costs', null=True)
    project_from = models.CharField(max_length=255, null=True)
    project_to = models.CharField(max_length=255, null=True)
    cost_value = models.IntegerField(default=0)

    def __str__(self):
        return f"Cost from {self.project_from} to {self.project_to} for {self.project.name}"

class Calendar(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='calendars', null=True)
    team_lead = models.CharField(max_length=255, null=True)
    date = models.DateField(null=True)
    value = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.team_lead} on {self.date} for {self.project.name}"

# Correct the signal to ensure it triggers after `Set` creation
@receiver(post_save, sender=Set)
def generate_related_data(sender, instance, created, **kwargs):
    if created:
        project = instance.project
        start_date = project.start_date
        finish_date = project.finish_date
        num_projects = instance.projects
        num_teamleads = instance.teamleads
        time_period = instance.timeperiod

        # Step 1: Generate Calendar Entries for each team lead
        delta = finish_date - start_date
        for lead_num in range(1, num_teamleads + 1):
            team_lead = f"TL{lead_num}"
            for day in range(delta.days + 1):
                date = start_date + timedelta(days=day)
                Calendar.objects.create(
                    project=project,
                    team_lead=team_lead,
                    date=date,
                    value=0  # Initially 0
                )

        # Step 2: Generate PreferenceCost entries
        project_names = [f"P{i+1}" for i in range(num_projects)]
        for row_project in project_names:
            for col_project in project_names:
                if row_project != col_project:
                    PreferenceCost.objects.create(
                        project=project,
                        project_from=row_project,
                        project_to=col_project,
                        cost_value=0
                    )

        # Step 3: Generate Reqskill Entries
        skill_names = [f"Skill{i+1}" for i in range(instance.skills)]
        for skill_name in skill_names:
            for proj_name in project_names:
                Reqskill.objects.create(
                    project=project,
                    skills=skill_name,
                    projects=proj_name
                )

        # Step 4: Generate Duration Entries
        for proj_name in project_names:
            duration_value = time_period
            Duration.objects.create(
                project=project,
                projects=proj_name
            )

        # Step 5: Generate MGskill, DS, FS, Score, Cost, and WAD
        manager_names = [f"Manager{i+1}" for i in range(instance.managers)]
        for skill_name in skill_names:
            for manager_name in manager_names:
                MGskill.objects.create(
                    project=project,
                    skills=skill_name,
                    managers=manager_name
                )

        for proj_name in project_names:
            DS.objects.create(
                project=project,
                projects=proj_name
            )
            FS.objects.create(
                project=project,
                projects=proj_name
            )

        for lead_num in range(1, num_teamleads + 1):
            team_lead = f"TL{lead_num}"
            Score.objects.create(
                project=project,
                teamleads=team_lead
            )
            Cost.objects.create(
                project=project,
                teamleads=team_lead
            )
            WAD.objects.create(
                project=project,
                teamleads=team_lead
            )
