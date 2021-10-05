from django.contrib import admin
from django.urls import path,include
from django.conf.urls import url
from .views import *
urlpatterns = [
    path('', home,name='home'),
    path('cleandata/', cleandata,name='clean_data'),
    path('classify/',classify, name= 'classify'),
    path('parameters/',parameter, name= 'parameter'),
    path('startmodel/',startmodel, name= 'start'),
    path('test_options/',testoptions, name= 'testoptions'),
    path('result/',result, name= 'result'),
    path('CreateModel/',CreatModel,name='Create'),
    path('Test/',Test,name='Test'),
    path('Clean/',Clean,name='clean'),


]
