from django.contrib import admin

# Register your models here.
from .models import Skill, Duration, Set, Project,Calendar,PreferenceCost, Cost, Score, WAD,FS, DS, MGskill, Reqskill, ProjectFiles

admin.site.register(Skill)
admin.site.register(Duration)
admin.site.register(Set)
admin.site.register(Project)
admin.site.register(Calendar)
admin.site.register(PreferenceCost)
admin.site.register(Cost)
admin.site.register(Score)
admin.site.register(WAD)
admin.site.register(DS)
admin.site.register(FS)
admin.site.register(MGskill)
admin.site.register(Reqskill)
admin.site.register(ProjectFiles)