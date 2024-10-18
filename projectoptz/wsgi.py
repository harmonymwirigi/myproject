"""
WSGI config for projectoptz project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/wsgi/
"""

import os
# importing whitenoise
from whitenoise import WhiteNoise

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'projectoptz.settings')

application = get_wsgi_application()
# wrapping up existing wsgi application
application = WhiteNoise(application, root="static")
