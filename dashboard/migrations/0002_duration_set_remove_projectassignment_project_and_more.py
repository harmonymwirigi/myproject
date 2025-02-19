# Generated by Django 5.1.1 on 2024-09-28 13:30

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Duration',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Projects', models.CharField(max_length=100, null=True)),
                ('Value', models.IntegerField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Set',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('projects', models.IntegerField(null=True)),
                ('Teamleads', models.IntegerField(null=True)),
                ('Timeperiod', models.IntegerField(null=True)),
                ('Skills', models.IntegerField(null=True)),
                ('Managers', models.IntegerField(null=True)),
            ],
        ),
        migrations.RemoveField(
            model_name='projectassignment',
            name='project',
        ),
        migrations.RemoveField(
            model_name='projectassignment',
            name='worker',
        ),
        migrations.RemoveField(
            model_name='projectrequirement',
            name='project',
        ),
        migrations.RemoveField(
            model_name='projectrequirement',
            name='skill',
        ),
        migrations.RemoveField(
            model_name='worker',
            name='skills',
        ),
        migrations.RemoveField(
            model_name='workerskillavailability',
            name='worker',
        ),
        migrations.RemoveField(
            model_name='workerskillavailability',
            name='skill',
        ),
        migrations.RemoveField(
            model_name='skill',
            name='name',
        ),
        migrations.AddField(
            model_name='skill',
            name='Skills',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='skill',
            name='Teamleads',
            field=models.CharField(max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='skill',
            name='Value',
            field=models.CharField(max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='project',
            name='duration',
            field=models.IntegerField(null=True),
        ),
        migrations.AlterField(
            model_name='project',
            name='finish_date',
            field=models.DateField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='project',
            name='start_date',
            field=models.DateField(default=django.utils.timezone.now),
        ),
        migrations.DeleteModel(
            name='OutputData',
        ),
        migrations.DeleteModel(
            name='ProjectAssignment',
        ),
        migrations.DeleteModel(
            name='ProjectRequirement',
        ),
        migrations.DeleteModel(
            name='Worker',
        ),
        migrations.DeleteModel(
            name='WorkerSkillAvailability',
        ),
    ]
