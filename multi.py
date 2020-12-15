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


def count_pixels(part,inverted_folder):
    import PIL.ImageOps
    from skimage import measure


    ##leer e invertir colores
    image = PILImage.open(part)
    inverted_image = PIL.ImageOps.invert(image)
    inverted_image.save(inverted_folder/ pathlib.Path(part).name)

    ##Leer y escala de grises
    file = inverted_folder / pathlib.Path(part).name
    image = cv2.imread(str(file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]

    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    # loop over the unique components
    pixels_count = []
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 0:
            pixels_count.append(numPixels)
            mask = cv2.add(mask, labelMask)
    return pixels_count


def brillo_contraste(root_folder,part,ct, br,save_folder):

    #read the image
    file = root_folder / pathlib.Path(part).name
    im = PILImage.open(file)

    #image brightness enhancer
    enhancer = ImageEnhance.Contrast(im)

    factor = ct #increase contrast
    im_output = enhancer.enhance(factor)
    #im_output.show()

    enhancer = ImageEnhance.Brightness(im_output)

    factor = br#brightens the image
    im_output2 = enhancer.enhance(factor)
    #im_output2.show()
    new_file = save_folder/ pathlib.Path(part).name
    im_output2.save(new_file)


def gauss_blur(root_folder, part, gs, bl, save_folder):

    # read image
    file = str(root_folder / pathlib.Path(part).name)
    src = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    # apply guassian blur on src image
    dst = cv2.GaussianBlur(src, (gs, gs), cv2.BORDER_DEFAULT)
    a_img = cv2.blur(dst, (bl, bl))
    dst1 = cv2.GaussianBlur(a_img, (gs, gs), cv2.BORDER_DEFAULT)

    im_saved = PILImage.fromarray(dst1, mode='RGB')
    new_file = save_folder / pathlib.Path(part).name
    im_saved.save(new_file)
    # display input and output image
    # cv2.imshow("Gaussian Smoothing",numpy.hstack((src,dst1)))
    # cv2.waitKey(0) # waits until a key is pressed
    # cv2.destroyAllWindows() # destroys the window showing image


def diferencia(folder_1, folder_2, part, save_folder):

    file1 = folder_1 / pathlib.Path(part).name
    file2 = folder_2 / pathlib.Path(part).name
    img1= PILImage.open(file1)
    img2= PILImage.open(file2)

    diff = ImageChops.difference(img1, img2)

    #diff.show()
    new_file = save_folder / pathlib.Path(part).name
    diff.save(new_file)


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class MainScreen(ScreenManager):
	pass


class SelectApp(Screen):
	pass


class CroppingApp(Screen):
	loadfile = ObjectProperty(None)
	text_input = ObjectProperty(None)

	def dismiss_popup(self):
		self._popup.dismiss()

	def show_load(self):
		content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
		self._popup = Popup(title="Load file", content=content,
							size_hint=(0.9, 0.9))
		self._popup.open()

	def load(self, path, filename):
		self.item_list = []
		for file in filename:
			self.item_list.append(file)

		if len(self.item_list) == 1:
			self.ids.crp_all_files_btn.disabled = False

		self.dismiss_popup()

	def on_pre_enter(self):
		self.ids.crp_all_files_btn.disabled = True

	def load_all_files(self):

		path = pathlib.Path(self.item_list[0])
		mypath = str(list(path.parents)[0])
		onlyfiles = [mypath + "\\" + f for f in listdir(mypath) if
					 (isfile(join(mypath, f)) and join(mypath, f)[-4:].lower() == ".jpg")]
		self.item_list = onlyfiles.copy()

	def test_cropping(self):
		try:

			path = pathlib.Path(self.item_list[0])
			parent = list(path.parents)[0] / 'Cropped'


			if parent.exists():
				pass
			else:
				parent.mkdir()
			# Opens a image in RGB mode
			part = self.item_list[0]
			im = PILImage.open(part)
			im_size = str(im.size)
			# Setting the points for cropped image
			left = int(self.ids.left_in.text)
			top = int(self.ids.top_in.text)
			right = int(self.ids.right_in.text)
			bottom = int(self.ids.bottom_in.text)
			# Cropped image of above dimension
			# (It will not change orginal image)
			im1 = im.crop((left, top, right, bottom))
			new_file = parent / pathlib.Path(part).name
			im1.save(new_file)
			self.ids.cropped.source = str(new_file)
			self.ids.lb_size.text = "Initial Size = "+im_size
			self.ids.cropped.reload()

		except (ValueError, SystemError):
			dialog = MDDialog(title="ERROR",
							  text="use correct pixel values "
							  )
			dialog.open()

		except (AttributeError):
			dialog = MDDialog(title="ERROR",
							  text="Select files first "
							  )
			dialog.open()

	def cropping_all(self):
		try:
			path = pathlib.Path(self.item_list[0])
			parent = list(path.parents)[0] / 'Cropped'

			if parent.exists():
				pass
			else:
				parent.mkdir()

			for file in self.item_list:
				# Opens a image in RGB mode
				part = file
				im = PILImage.open(part)
				# Setting the points for cropped image
				left = int(self.ids.left_in.text)
				top = int(self.ids.top_in.text)
				right = int(self.ids.right_in.text)
				bottom = int(self.ids.bottom_in.text)
				# Cropped image of above dimension
				# (It will not change orginal image)
				im1 = im.crop((left, top, right, bottom))
				new_file = parent / pathlib.Path(file).name
				im1.save(new_file)
			dialog = MDDialog(title="The Pictures are ready.",
							  text=f"Check {parent}"

							  )
			dialog.open()
		except (ValueError, SystemError):
			dialog = MDDialog(title="ERROR",
							  text="use correct pixel values "
							  )
			dialog.open()

		except (AttributeError):
			dialog = MDDialog(title="ERROR",
							  text="Select files first "
							  )
			dialog.open()


class RotationApp(Screen):
	loadfile = ObjectProperty(None)
	text_input = ObjectProperty(None)

	def dismiss_popup(self):
		self._popup.dismiss()

	def show_load(self):
		content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
		self._popup = Popup(title="Load file", content=content,
							size_hint=(0.9, 0.9))
		self._popup.open()

	def load(self, path, filename):
		self.item_list = []
		for file in filename:
			self.item_list.append(file)

		if len(self.item_list) == 1:
			self.ids.rot_all_files_btn.disabled = False

		self.dismiss_popup()

	def on_pre_enter(self):
		self.ids.rot_all_files_btn.disabled = True

	def load_all_files(self):

		path = pathlib.Path(self.item_list[0])
		mypath = str(list(path.parents)[0])
		onlyfiles = [mypath + "\\" + f for f in listdir(mypath) if
					 (isfile(join(mypath, f)) and join(mypath, f)[-4:].lower() == ".jpg")]
		self.item_list = onlyfiles.copy()

	def test_rotation(self):
		try:
			# Opens a image in RGB mode
			part = self.item_list[0]
			im = cv2.imread(part)
			angle = float(self.ids.angle_in.text)
			rotated = imutils.rotate(im, angle)
			rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
			image_sav = PILImage.fromarray(rotated)
			image_sav.show()
		except (ValueError, SystemError):
			dialog = MDDialog(title="ERROR",
							  text="use correct angle value "
							  )
			dialog.open()

		except (AttributeError):
			dialog = MDDialog(title="ERROR",
							  text="Select files first "
							  )
			dialog.open()

	def rotation_all(self):
		try:
			path = pathlib.Path(self.item_list[0])
			parent = list(path.parents)[0] / 'Rotated'

			if parent.exists():
				pass
			else:
				parent.mkdir()

			for file in self.item_list:
				# Opens a image in RGB mode
				part = file
				im = cv2.imread(part)
				angle = float(self.ids.angle_in.text)
				rotated = imutils.rotate(im, angle)
				rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
				image_sav = PILImage.fromarray(rotated)
				new_file = parent / pathlib.Path(file).name
				image_sav.save(new_file)
			dialog = MDDialog(title="The Pictures are ready.",
							  text=f"Check {parent}"

							  )
			dialog.open()
		except (ValueError, SystemError):
			dialog = MDDialog(title="ERROR",
							  text="use correct angle value "
							  )
			dialog.open()

		except (AttributeError):
			dialog = MDDialog(title="ERROR",
							  text="Select files first "
							  )
			dialog.open()


class GaussBlurApp(Screen):
	loadfile = ObjectProperty(None)
	text_input = ObjectProperty(None)

	def dismiss_popup(self):
		self._popup.dismiss()

	def show_load(self):
		content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
		self._popup = Popup(title="Load file", content=content,
							size_hint=(0.9, 0.9))
		self._popup.open()

	def load(self, path, filename):
		self.item_list = []
		for file in filename:
			self.item_list.append(file)

		if len(self.item_list) == 1:
			self.ids.gb_all_files_btn.disabled = False

		self.dismiss_popup()

	def on_pre_enter(self):
		self.ids.gb_all_files_btn.disabled = True

	def load_all_files(self):

		path = pathlib.Path(self.item_list[0])
		mypath = str(list(path.parents)[0])
		onlyfiles = [mypath + "\\" + f for f in listdir(mypath) if
					 (isfile(join(mypath, f)) and join(mypath, f)[-4:].lower() == ".jpg")]
		self.item_list = onlyfiles.copy()

	def test_filters(self):
		try:
			path = pathlib.Path(self.item_list[0])

			folders_list = [list(path.parents)[0] / 'Filtered',
							list(path.parents)[0] / 'Filtered/DELETE_ME',
							list(path.parents)[0] / 'Filtered/DELETE_ME/c_b',
							list(path.parents)[0] / 'Filtered/DELETE_ME/gb_c_b',
							list(path.parents)[0] / 'Filtered/DELETE_ME/mm_gb_c_b',
							list(path.parents)[0] / 'Filtered/DELETE_ME/diff']

			for folder in folders_list:
				if folder.exists():
					pass
				else:
					folder.mkdir()

			ct1 = float(self.ids.contrast_in.text)
			br1 = float(self.ids.brightness_in.text)
			gs1 = int(self.ids.gauss_in.text)
			bl1 = int(self.ids.blur_in.text)
			ct2 = float(self.ids.contrast2_in.text)
			br2 = float(self.ids.brightness2_in.text)
			gs2 = int(self.ids.gauss2_in.text)
			bl2 = int(self.ids.blur2_in.text)
			part = self.item_list[0]

			brillo_contraste(list(path.parents)[0], part, ct1, br1, folders_list[2])
			gauss_blur(folders_list[2], part, gs1, bl1, folders_list[3])
			brillo_contraste(folders_list[3], part, ct2, br2, folders_list[4])
			diferencia(folders_list[2], folders_list[4], part, folders_list[5])
			gauss_blur(folders_list[5], part, gs2, bl2, folders_list[0])

			im = PILImage.open(folders_list[0] / pathlib.Path(part).name)
			im.show()

		except (ValueError, SystemError):
			dialog = MDDialog(title="ERROR",
							  text="use correct values "
							  )
			dialog.open()

		except (AttributeError):
			dialog = MDDialog(title="ERROR",
							  text="Select files first "
							  )
			dialog.open()

	def filters_all(self):

		try:
			path = pathlib.Path(self.item_list[0])

			folders_list = [list(path.parents)[0] / 'Filtered',
							list(path.parents)[0] / 'Filtered/DELETE_ME',
							list(path.parents)[0] / 'Filtered/DELETE_ME/c_b',
							list(path.parents)[0] / 'Filtered/DELETE_ME/gb_c_b',
							list(path.parents)[0] / 'Filtered/DELETE_ME/mm_gb_c_b',
							list(path.parents)[0] / 'Filtered/DELETE_ME/diff']

			for folder in folders_list:
				if folder.exists():
					pass
				else:
					folder.mkdir()

			for file in self.item_list:
				ct1 = float(self.ids.contrast_in.text)
				br1 = float(self.ids.brightness_in.text)
				gs1 = int(self.ids.gauss_in.text)
				bl1 = int(self.ids.blur_in.text)
				ct2 = float(self.ids.contrast2_in.text)
				br2 = float(self.ids.brightness2_in.text)
				gs2 = int(self.ids.gauss2_in.text)
				bl2 = int(self.ids.blur2_in.text)
				part = file

				brillo_contraste(list(path.parents)[0], part, ct1, br1, folders_list[2])
				gauss_blur(folders_list[2], part, gs1, bl1, folders_list[3])
				brillo_contraste(folders_list[3], part, ct2, br2, folders_list[4])
				diferencia(folders_list[2], folders_list[4], part, folders_list[5])
				gauss_blur(folders_list[5], part, gs2, bl2, folders_list[0])

			dialog = MDDialog(title="The Pictures are ready.",
							  text=f"Check {folders_list[0]}"
							  )
			dialog.open()
		except (ValueError, SystemError):
			dialog = MDDialog(title="ERROR",
							  text="use correct values "
							  )
			dialog.open()

		except (AttributeError):
			dialog = MDDialog(title="ERROR",
							  text="Select files first "
							  )
			dialog.open()


class PixelsCountApp(Screen):
	loadfile = ObjectProperty(None)
	text_input = ObjectProperty(None)

	def dismiss_popup(self):
		self._popup.dismiss()

	def show_load(self):
		content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
		self._popup = Popup(title="Load file", content=content,
							size_hint=(0.9, 0.9))
		self._popup.open()

	def load(self, path, filename):
		self.item_list = []
		for file in filename:
			self.item_list.append(file)

		if len(self.item_list) == 1:
			self.ids.count_all_files_btn.disabled = False

		self.dismiss_popup()

	def on_pre_enter(self):
		self.ids.count_all_files_btn.disabled = True

	def load_all_files(self):

		path = pathlib.Path(self.item_list[0])
		mypath = str(list(path.parents)[0])
		onlyfiles = [mypath + "\\" + f for f in listdir(mypath) if
					 (isfile(join(mypath, f)) and join(mypath, f)[-4:].lower() == ".jpg")]
		self.item_list = onlyfiles.copy()



	def count_all(self):




		try:

			path = pathlib.Path(self.item_list[0])
			folders_list = [list(path.parents)[0] / 'Pixel_count',
							list(path.parents)[0] / 'Pixel_count/DELETE_ME']

			for folder in folders_list:
				if folder.exists():
					pass
				else:
					folder.mkdir()

			matriz_partes = []

			for file in self.item_list:
				image = cv2.imread(file)
				count = count_pixels(file, folders_list[1])
				count.append(image.shape[:3])
				count.append(pathlib.Path(file).name)
				matriz_partes.append(count[::-1])

			excel_file = list(path.parents)[0] / 'Pixel_count/Pixel_count_.xlsx'
			data = pd.DataFrame(matriz_partes)
			datatoexcel = pd.ExcelWriter(excel_file, engine='xlsxwriter')
			data.to_excel(datatoexcel, sheet_name='Sheet1', header=False, index=False)
			datatoexcel.save()

			dialog = MDDialog(title="The Pictures are ready.",
							  text=f"Check {folders_list[0]}"
							  )
			dialog.open()

		except AttributeError:
			dialog = MDDialog(title="ERROR",
			                  text="Select files first "
			)
			dialog.open()


class ThresholdApp(Screen):
	loadfile = ObjectProperty(None)
	text_input = ObjectProperty(None)

	def on_pre_enter(self):
		self.ids.thrs_all_files_btn.disabled = True


	def dismiss_popup(self):
		self._popup.dismiss()

	def show_load(self):
		content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
		self._popup = Popup(title="Load file", content=content,
							size_hint=(0.9, 0.9))
		self._popup.open()

	def load(self, path, filename):
		self.item_list = []
		self.picture_n = 0
		for file in filename:
			self.item_list.append(file)


		if len(self.item_list) == 1:
			self.ids.thrs_all_files_btn.disabled = False

		self.dismiss_popup()

	def load_all_files (self):

		path = pathlib.Path(self.item_list[0])
		mypath = str(list(path.parents)[0])
		onlyfiles = [mypath + "\\" + f for f in listdir(mypath) if (isfile(join(mypath, f)) and join(mypath, f)[-4:].lower() == ".jpg")]
		self.item_list = onlyfiles.copy()
		if self.picture_n >= len(self.item_list):
			self.ids.bt_next.disabled = True
		else:
			self.ids.bt_next.disabled = False


	def show_picture(self,threshold_value):

		try:
			path = pathlib.Path(self.item_list[self.picture_n])
			parent = list(path.parents)[0] / 'threshold'

			self.ids.bt_next.disabled = False

			if parent.exists():
				pass
			else:
				parent.mkdir()

			if self.picture_n + 1 >= len(self.item_list):
				self.ids.bt_next.disabled = True

			# Opens a image in RGB mode
			part = self.item_list[self.picture_n]
			compare = list(path.parents)[0] / 'compare from'
			original_part = compare / pathlib.Path(part).name
			if (not(compare.exists()) or not(original_part.exists())):
				dialog = MDDialog(title="ERROR ORIGINAL FILES MISSING",
								  text="* Make sure the 'compare from' folder exists\n* Make sure the files to compare from are in the folder "
								  )
				dialog.open()

			self.ids.original.source = str(original_part)
			self.ids.title.text = str(pathlib.Path(part).name)
			self.ids.original.reload()

			img_threshold = cv2.imread(part, 0)
			ret, thresh1 = cv2.threshold(img_threshold, threshold_value, 255, cv2.THRESH_BINARY)
			image_sav = PILImage.fromarray(thresh1)
			new_file = parent / pathlib.Path(part).name
			image_sav.save(new_file)

			self.ids.threshold.source = str(new_file)
			self.ids.threshold.reload()

		except(AttributeError):
			dialog = MDDialog(title="ERROR",
							  text="NO FILES LOADED "
							  )
			dialog.open()

	def new_threshold(self, threshold_value):

		try:
			path = pathlib.Path(self.item_list[self.picture_n])
			parent = list(path.parents)[0] / 'threshold'
			part = self.item_list[self.picture_n]
			compare = list(path.parents)[0] / 'compare from'
			original_part = compare / pathlib.Path(part).name
			if (not (compare.exists()) or not (original_part.exists())):
				dialog = MDDialog(title="ERROR ORIGINAL FILES MISSING",
								  text="* Make sure the 'compare from' folder exists\n* Make sure the files to compare from are in the folder "
								  )
				dialog.open()
			self.ids.original.source = str(original_part)

			img_threshold = cv2.imread(part, 0)
			ret, thresh1 = cv2.threshold(img_threshold, threshold_value, 255, cv2.THRESH_BINARY)
			image_sav = PILImage.fromarray(thresh1)
			new_file = parent / pathlib.Path(part).name
			image_sav.save(new_file)

			self.ids.threshold.source = str(new_file)
			self.ids.original.reload()
			self.ids.threshold.reload()

		except(AttributeError):
			dialog = MDDialog(title="ERROR",
							  text="NO FILES LOADED "
							  )
			dialog.open()

	def next_picture(self,threshold_value):

		try:
			self.picture_n += 1
			self.ids.bt_next.disabled = False

			path = pathlib.Path(self.item_list[self.picture_n])
			parent = list(path.parents)[0] / 'threshold'
			# Opens a image in RGB mode
			part = self.item_list[self.picture_n]
			compare = list(path.parents)[0] / 'compare from'
			original_part = compare / pathlib.Path(part).name
			if (not (compare.exists()) or not (original_part.exists())):
				dialog = MDDialog(title="ERROR ORIGINAL FILES MISSING",
								  text="* Make sure the 'compare from' folder exists\n* Make sure the files to compare from are in the folder "
								  )
				dialog.open()
			self.ids.original.source = str(original_part)
			self.ids.original.reload()

			img_threshold = cv2.imread(part, 0)
			ret, thresh1 = cv2.threshold(img_threshold, threshold_value, 255, cv2.THRESH_BINARY)
			image_sav = PILImage.fromarray(thresh1)
			new_file = parent / pathlib.Path(part).name
			image_sav.save(new_file)

			self.ids.threshold.source = str(new_file)
			self.ids.threshold.reload()
			self.ids.title.text = str(pathlib.Path(part).name)

			if self.picture_n + 1 >= len(self.item_list):
				self.ids.bt_next.disabled = True


		except(AttributeError):
			dialog = MDDialog(title="ERROR",
							  text="NO FILES LOADED "
							  )
			dialog.open()


class DivideApp(Screen):
	loadfile = ObjectProperty(None)
	text_input = ObjectProperty(None)

	def dismiss_popup(self):
		self._popup.dismiss()

	def show_load(self):
		content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
		self._popup = Popup(title="Load file", content=content,
							size_hint=(0.9, 0.9))
		self._popup.open()

	def load(self, path, filename):
		self.item_list = []
		for file in filename:
			self.item_list.append(file)

		if len(self.item_list) == 1:
			self.ids.div_all_files_btn.disabled = False

		self.dismiss_popup()

	def on_pre_enter(self):
		self.ids.div_all_files_btn.disabled = True

	def load_all_files (self):

		path = pathlib.Path(self.item_list[0])
		mypath = str(list(path.parents)[0])
		onlyfiles = [mypath + "\\" + f for f in listdir(mypath) if (isfile(join(mypath, f)) and join(mypath, f)[-4:].lower() == ".jpg")]
		self.item_list = onlyfiles.copy()

	def test_divide(self):
		try:
			path = pathlib.Path(self.item_list[0])

			folders_list = [list(path.parents)[0] / 'Divided',
							list(path.parents)[0] / 'Divided/DELETE_ME',]


			for folder in folders_list:
				if folder.exists():
					pass
				else:
					folder.mkdir()

			ct1 = float(self.ids.div_contrast_in.text)
			part = self.item_list[0]

			img = PILImage.open(part)
			img_inverted = ImageOps.invert(img)

			enhancer = ImageEnhance.Contrast(img_inverted)
			im_output = enhancer.enhance(ct1)
			im_output.save(folders_list[1] / pathlib.Path(part).name)

			imgA = img
			imgA.load()
			imgB = PILImage.open(folders_list[1] / pathlib.Path(part).name)
			imgB.load()

			# split RGB images into 3 channels
			rA, gA, bA = imgA.split()
			rB, gB, bB = imgB.split()

			# divide each channel (image1/image2)
			rTmp = ImageMath.eval("int(a/((float(b)+1)/256))", a=rA, b=rB).convert('L')
			gTmp = ImageMath.eval("int(a/((float(b)+1)/256))", a=gA, b=gB).convert('L')
			bTmp = ImageMath.eval("int(a/((float(b)+1)/256))", a=bA, b=bB).convert('L')

			# merge channels into RGB image
			imgOut = PILImage.merge("RGB", (rTmp, gTmp, bTmp))

			imgOut.save(folders_list[0] / pathlib.Path(part).name)




			gs1 = int(self.ids.div_gauss.text)
			bl1 = int(self.ids.div_blur.text)


			gauss_blur(folders_list[0], part, gs1, bl1, folders_list[0])


			im = PILImage.open(folders_list[0] / pathlib.Path(part).name)
			im.show()





		except (ValueError, SystemError):
			dialog = MDDialog(title="ERROR",
							  text="use correct values "
							  )
			dialog.open()

		except (AttributeError):
			dialog = MDDialog(title="ERROR",
							  text="Select files first "
							  )
			dialog.open()

	def divide_all(self):

		try:

			path = pathlib.Path(self.item_list[0])

			folders_list = [list(path.parents)[0] / 'Divided',
							list(path.parents)[0] / 'Divided/DELETE_ME', ]

			for folder in folders_list:
				if folder.exists():
					pass
				else:
					folder.mkdir()

			for file in self.item_list:

				ct1 = float(self.ids.div_contrast_in.text)
				part = file

				img = PILImage.open(part)
				img_inverted = ImageOps.invert(img)

				enhancer = ImageEnhance.Contrast(img_inverted)
				im_output = enhancer.enhance(ct1)
				im_output.save(folders_list[1] / pathlib.Path(part).name)

				imgA = img
				imgA.load()
				imgB = PILImage.open(folders_list[1] / pathlib.Path(part).name)
				imgB.load()

				# split RGB images into 3 channels
				rA, gA, bA = imgA.split()
				rB, gB, bB = imgB.split()

				# divide each channel (image1/image2)
				rTmp = ImageMath.eval("int(a/((float(b)+1)/256))", a=rA, b=rB).convert('L')
				gTmp = ImageMath.eval("int(a/((float(b)+1)/256))", a=gA, b=gB).convert('L')
				bTmp = ImageMath.eval("int(a/((float(b)+1)/256))", a=bA, b=bB).convert('L')

				# merge channels into RGB image
				imgOut = PILImage.merge("RGB", (rTmp, gTmp, bTmp))

				imgOut.save(folders_list[0] / pathlib.Path(part).name)

				gs1 = int(self.ids.div_gauss.text)
				bl1 = int(self.ids.div_blur.text)

				gauss_blur(folders_list[0], part, gs1, bl1, folders_list[0])


			dialog = MDDialog(title="The Pictures are ready.",
							  text=f"Check {folders_list[0]}"
							  )
			dialog.open()
		except (ValueError, SystemError):
			dialog = MDDialog(title="ERROR",
							  text="use correct values "
							  )
			dialog.open()

		except (AttributeError):
			dialog = MDDialog(title="ERROR",
							  text="Select files first "
							  )
			dialog.open()

class MultiApp(MDApp):

	def build(self):
		self.theme_cls.theme_style = "Dark"
		return MainScreen()

if __name__ == '__main__':

	app = MultiApp()
	app.run()