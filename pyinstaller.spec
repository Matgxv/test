# -*- mode: python ; coding: utf-8 -*-

from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen, ScreenManager


from os import listdir
from os.path import isfile, join
from PIL import Image as PILImage
from PIL import ImageEnhance, ImageChops, ImageMath, ImageOps
import pathlib
import imutils
import cv2
import numpy as np
import pandas as pd


from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivymd.uix.dialog import MDDialog


import os

from kivy.tools.packaging import pyinstaller_hooks as hooks
from kivymd import hooks_path as kivymd_hooks_path
from kivy_deps import sdl2, glew, gstreamer

""" 
   Next make sure you have all of kivys dependencies 
"""

block_cipher = None
kivy_deps_all = hooks.get_deps_all()
kivy_factory_modules = hooks.get_factory_modules()

path = os.path.abspath("D:\Manuel\Pycharm_projects\DATEC\Samples\multi_app")
path_data = os.path.abspath("D:\Manuel\Pycharm_projects\DATEC\Samples\multi_app\Resources")

# list of modules to exclude from analysis
excludes_a = ['Tkinter', '_tkinter', 'twisted', 'docutils', 'pygments']

# list of hiddenimports
hiddenimports1 = kivy_deps_all['hiddenimports'] + kivy_factory_modules + ['win32timezone'] + ['pkg_resources.py2_warn']

# binary data
sdl2_bin_tocs = [Tree(p) for p in sdl2.dep_bins]
glew_bin_tocs = [Tree(p) for p in glew.dep_bins]
gstreamer_bin_tocs = [Tree(p) for p in gstreamer.dep_bins]
bin_tocs = sdl2_bin_tocs + glew_bin_tocs + gstreamer_bin_tocs

# assets
# kivy_assets_toc = Tree(kivy_data_dir, prefix=join('kivy_install', 'data'))
# source_assets_toc = Tree('images', prefix='images')
# assets_toc = [kivy_assets_toc, source_assets_toc]

tocs = bin_tocs

a = Analysis(
    ["multi.py"],
    pathex=[path],
    datas=[(path_data,"Resources")],
    binaries=[],
    hiddenimports= kivy_deps_all['hiddenimports'] + kivy_factory_modules + ['win32timezone'] +['pkg_resources.py2_warn'],
    hookspath=[kivymd_hooks_path,"D:\Manuel\Pycharm_projects\DATEC\Samples\multi_app"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    *tocs,
    debug=False,
    strip=False,
    upx=True,
    name="Image_Processing",
    console=True,
    )