import os

DEBUG = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_PREFIX = '/static/'
STATIC_DIR = os.path.join(BASE_DIR, 'dist')

if DEBUG:
    STATIC_DIR = os.path.join(BASE_DIR, 'static')

