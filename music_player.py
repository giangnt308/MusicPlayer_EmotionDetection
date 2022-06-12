'''
Documentation, License etc.

@package music_player
'''
from winsound import PlaySound
import mido
import sys
from os.path import expanduser
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtCore import *

import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

import tensorflow.compat.v1 as tf
import numpy as np
import tensorflow as tf
import midi_statistics
import utils
import os
from gensim.models import Word2Vec
from playsound import playsound
from pygame import mixer
import time


class MainWindow(QMainWindow):
	sound = 0
	def __init__(self):
		super().__init__()
	
		#self.currentFile = '/'
		self.currentPlaylist = QMediaPlaylist()
		self.player = QMediaPlayer()
		self.userAction = -1			#0- stopped, 1- playing 2-paused
		self.player.mediaStatusChanged.connect(self.qmp_mediaStatusChanged)
		self.player.stateChanged.connect(self.qmp_stateChanged)
		self.player.positionChanged.connect(self.qmp_positionChanged)
		self.player.volumeChanged.connect(self.qmp_volumeChanged)
		self.player.setVolume(60)
		#Add Status bar
		self.statusBar().showMessage('No Media :: %d'%self.player.volume())
		self.homeScreen()
		
	def homeScreen(self):
		#Set title of the MainWindow
		self.setWindowTitle('Music Player')
		
		#Create Menubar
		self.createMenubar()
		
		#Create Toolbar
		self.createToolbar()
		
		#Add info screen
		#infoscreen = self.createInfoScreen()
		
		#Add Control Bar
		controlBar = self.addControls()
		
		#need to add both infoscreen and control bar to the central widget.
		centralWidget = QWidget()
		centralWidget.setLayout(controlBar)
		self.setCentralWidget(centralWidget)
		
		#Set Dimensions of the MainWindow
		self.resize(200,160)
		
		#show everything.
		self.show()
		
	def createMenubar(self):
		menubar = self.menuBar()
		filemenu = menubar.addMenu('File')
		filemenu.addAction(self.fileOpen())
		filemenu.addAction(self.songInfo())
		filemenu.addAction(self.folderOpen())
		filemenu.addAction(self.emotionDetection())
		filemenu.addAction(self.musicGenerate())
		filemenu.addAction(self.exitAction())
		

		
	def createToolbar(self):
		pass
	
	def addControls(self):
		controlArea = QVBoxLayout()		#centralWidget
		seekSliderLayout = QHBoxLayout()
		controls = QHBoxLayout()
		playlistCtrlLayout = QHBoxLayout()
		
		#creating buttons
		playBtn = QPushButton('Play')		#play button
		pauseBtn = QPushButton('Pause')		#pause button
		stopBtn = QPushButton('Stop')		#stop button
		volumeDescBtn = QPushButton('V (-)')#Decrease Volume
		volumeIncBtn = QPushButton('V (+)')	#Increase Volume
		
		#creating playlist controls
		prevBtn = QPushButton('Prev Song')
		nextBtn = QPushButton('Next Song')
		
		#creating seek slider
		seekSlider = QSlider()
		seekSlider.setMinimum(0)
		seekSlider.setMaximum(100)
		seekSlider.setOrientation(Qt.Horizontal)
		seekSlider.setTracking(False)
		seekSlider.sliderMoved.connect(self.seekPosition)
		#seekSlider.valueChanged.connect(self.seekPosition)
		
		seekSliderLabel1 = QLabel('0.00')
		seekSliderLabel2 = QLabel('0.00')
		seekSliderLayout.addWidget(seekSliderLabel1)
		seekSliderLayout.addWidget(seekSlider)
		seekSliderLayout.addWidget(seekSliderLabel2)
		
		#Add handler for each button. Not using the default slots.
		playBtn.clicked.connect(self.playHandler)
		pauseBtn.clicked.connect(self.pauseHandler)
		stopBtn.clicked.connect(self.stopHandler)
		volumeDescBtn.clicked.connect(self.decreaseVolume)
		volumeIncBtn.clicked.connect(self.increaseVolume)
		
		#Adding to the horizontal layout
		controls.addWidget(volumeDescBtn)
		controls.addWidget(playBtn)
		controls.addWidget(pauseBtn)
		controls.addWidget(stopBtn)
		controls.addWidget(volumeIncBtn)
		
		#playlist control button handlers
		prevBtn.clicked.connect(self.prevItemPlaylist)
		nextBtn.clicked.connect(self.nextItemPlaylist)
		playlistCtrlLayout.addWidget(prevBtn)
		playlistCtrlLayout.addWidget(nextBtn)
		
		#Adding to the vertical layout
		controlArea.addLayout(seekSliderLayout)
		controlArea.addLayout(controls)
		controlArea.addLayout(playlistCtrlLayout)
		return controlArea
	
	def playHandler(self):
		self.userAction = 1
		self.statusBar().showMessage('Playing at Volume %d'%self.player.volume())
		if self.player.state() == QMediaPlayer.StoppedState :
			if self.player.mediaStatus() == QMediaPlayer.NoMedia:
				#self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.currentFile)))
				print(self.currentPlaylist.mediaCount())
				if self.currentPlaylist.mediaCount() == 0:
					self.openFile()
				if self.currentPlaylist.mediaCount() != 0:
					self.player.setPlaylist(self.currentPlaylist)
			elif self.player.mediaStatus() == QMediaPlayer.LoadedMedia:
				self.player.play()
			elif self.player.mediaStatus() == QMediaPlayer.BufferedMedia:
				self.player.play()
		elif self.player.state() == QMediaPlayer.PlayingState:
			pass
		elif self.player.state() == QMediaPlayer.PausedState:
			self.player.play()
			
	def pauseHandler(self):
		self.userAction = 2
		self.statusBar().showMessage('Paused %s at position %s at Volume %d'%\
			(self.player.metaData(QMediaMetaData.Title),\
				self.centralWidget().layout().itemAt(0).layout().itemAt(0).widget().text(),\
					self.player.volume()))
		self.player.pause()
			
	def stopHandler(self):
		self.userAction = 0
		self.statusBar().showMessage('Stopped at Volume %d'%(self.player.volume()))
		if self.player.state() == QMediaPlayer.PlayingState:
			self.stopState = True
			self.player.stop()
		elif self.player.state() == QMediaPlayer.PausedState:
			self.player.stop()
		elif self.player.state() == QMediaPlayer.StoppedState:
			pass
		
	def qmp_mediaStatusChanged(self):
		if self.player.mediaStatus() == QMediaPlayer.LoadedMedia and self.userAction == 1:
			durationT = self.player.duration()
			self.centralWidget().layout().itemAt(0).layout().itemAt(1).widget().setRange(0,durationT)
			self.centralWidget().layout().itemAt(0).layout().itemAt(2).widget().setText('%d:%02d'%(int(durationT/60000),int((durationT/1000)%60)))
			self.player.play()
			
	def qmp_stateChanged(self):
		if self.player.state() == QMediaPlayer.StoppedState:
			self.player.stop()
			
	def qmp_positionChanged(self, position,senderType=False):
		sliderLayout = self.centralWidget().layout().itemAt(0).layout()
		if senderType == False:
			sliderLayout.itemAt(1).widget().setValue(position)
		#update the text label
		sliderLayout.itemAt(0).widget().setText('%d:%02d'%(int(position/60000),int((position/1000)%60)))
	
	def seekPosition(self, position):
		sender = self.sender()
		if isinstance(sender,QSlider):
			if self.player.isSeekable():
				self.player.setPosition(position)
				
	def qmp_volumeChanged(self):
		msg = self.statusBar().currentMessage()
		msg = msg[:-2] + str(self.player.volume())
		self.statusBar().showMessage(msg)
		
	def increaseVolume(self):
		vol = self.player.volume()
		vol = min(vol+5,100)
		self.player.setVolume(vol)
		
	def decreaseVolume(self):
		vol = self.player.volume()
		vol = max(vol-5,0)
		self.player.setVolume(vol)
	
	def fileOpen(self):
		fileAc = QAction('Open File',self)
		fileAc.setShortcut('Ctrl+O')
		fileAc.setStatusTip('Open File')
		fileAc.triggered.connect(self.openFile)
		return fileAc
		
	def openFile(self):
		fileChoosen = QFileDialog.getOpenFileUrl(self,'Open Music File')
		if fileChoosen != None:
			self.currentPlaylist.addMedia(QMediaContent(fileChoosen[0]))
	
	def folderOpen(self):
		folderAc = QAction('Open Folder',self)
		folderAc.setShortcut('Ctrl+D')
		folderAc.setStatusTip('Open Folder (Will add all the files in the folder) ')
		folderAc.triggered.connect(self.addFiles)
		return folderAc
	
	def addFiles(self):
		folderChoosen = QFileDialog.getExistingDirectory(self,'Open Music Folder', expanduser('~'))
		if folderChoosen != None:
			it = QDirIterator(folderChoosen)
			it.next()
			while it.hasNext():
				if it.fileInfo().isDir() == False and it.filePath() != '.':
					fInfo = it.fileInfo()
					print(it.filePath(),fInfo.suffix())
					if fInfo.suffix() in ('mp3','ogg','wav','mid'):
						print('added file ',fInfo.fileName())
						self.currentPlaylist.addMedia(QMediaContent(QUrl.fromLocalFile(it.filePath())))
				it.next()
			
	def songInfo(self):
		infoAc = QAction('Info',self)
		infoAc.setShortcut('Ctrl+I')
		infoAc.setStatusTip('Displays Current Song Information')
		infoAc.triggered.connect(self.displaySongInfo)
		return infoAc

	def displaySongInfo(self):
		metaDataKeyList = self.player.availableMetaData()
		fullText = '<table class="tftable" border="0">'
		for key in metaDataKeyList:
			value = self.player.metaData(key)
			fullText = fullText + '<tr><td>' + key + '</td><td>' + str(value) + '</td></tr>'
		fullText = fullText + '</table>'
		infoBox = QMessageBox(self)
		infoBox.setWindowTitle('Detailed Song Information')
		infoBox.setTextFormat(Qt.RichText)
		infoBox.setText(fullText)
		infoBox.addButton('OK',QMessageBox.AcceptRole)
		infoBox.show()
	
	def prevItemPlaylist(self):
		self.player.playlist().previous()
	
	def nextItemPlaylist(self):
		self.player.playlist().next()
	
	def emotionDetection(self):
		emotionDt = QAction('Emotion Detection',self)
		emotionDt.setShortcut('Ctrl+E')
		emotionDt.triggered.connect(self.emotion_detection)
		return emotionDt


	def emotion_detection(self):
		model = model_from_json(open("model_main.json", "r").read())
		model.load_weights('model_weights_main.h5')

			# prevents openCL usage and unnecessary logging messages
		cv2.ocl.setUseOpenCL(False)

			# dictionary which assigns each label an emotion (alphabetical order)
		emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

			# start the webcam feed
		cap = cv2.VideoCapture(0)
		while True:
				# Find haar cascade to draw bounding box around face
			ret, frame = cap.read()
			if not ret:
				break
			facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

			for (x, y, w, h) in faces:
				cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
				roi_gray = gray[y:y + h, x:x + w]
				cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
				prediction = model.predict(cropped_img)
				maxindex = int(np.argmax(prediction))
				cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
			cv2.imshow('Video', cv2.resize(frame,(400,400),interpolation = cv2.INTER_CUBIC))
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			for (x, y, w, h) in faces:
				cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
				roi_gray = gray[y:y + h, x:x + w]
				cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
				prediction = model.predict(cropped_img)
				maxindex = int(np.argmax(prediction))
				if maxindex == 0:
					#init
					mixer.init()
					mixer.music.load('./Music/emotional-piano-sad-background-music-for-videos-5688-angry.mp3') # you may use .mp3 but support is limited
					mixer.music.play()
					time.sleep(5)
					#playsound('./Music/emotional-piano-sad-background-music-for-videos-5688-angry.mp3')
				if maxindex == 1:
					#init
					mixer.init()
					mixer.music.load('./Music/easy-country-music-intro-outro-11915-disgusted.mp3') # you may use .mp3 but support is limited
					mixer.music.play()
					time.sleep(5)
					#playsound('./Music/easy-country-music-intro-outro-11915-disgusted.mp3')
				if maxindex == 2:
					#init
					mixer.init()
					mixer.music.load('./Music/relaxed-inspiration-ig-version-short-60s-6580-fearful.mp3') # you may use .mp3 but support is limited
					mixer.music.play()
					time.sleep(5)
					#playsound('./Music/relaxed-inspiration-ig-version-short-60s-6580-fearful.mp3')
				if maxindex == 3:
					#init
					mixer.init()
					mixer.music.load('./Music/cinematic-fairy-tale-story-short-kikc-8698-happy.mp3') # you may use .mp3 but support is limited
					mixer.music.play()
					time.sleep(5)
					#playsound('./Music/cinematic-fairy-tale-story-short-kikc-8698-happy.mp3')
				if maxindex == 4:
					#init
					mixer.init()
					mixer.music.load('./Music/inspiring-cinematic-uplifting-piano-short-8701-neutral.mp3') # you may use .mp3 but support is limited
					mixer.music.play()
					time.sleep(5)
					#playsound('./Music/inspiring-cinematic-uplifting-piano-short-8701-neutral.mp3')
				if maxindex == 5:
					#init
					mixer.init()
					mixer.music.load('./Music/acoustic-guitar-dream-30-seconds-4642-sad.mp3') # you may use .mp3 but support is limited
					mixer.music.play()
					time.sleep(5)
					#playsound('./Music/acoustic-guitar-dream-30-seconds-4642-sad.mp3')
				if maxindex == 6:
					#init
					mixer.init()
					mixer.music.load('./Music/FadedPianoCover-VA-4405663-Surprised.mp3') # you may use .mp3 but support is limited
					mixer.music.play()
					time.sleep(5)
					#playsound('./Music/FadedPianoCover-VA-4405663-Surprised.mp3')
		cap.release()
		cv2.destroyAllWindows()
		

	def musicGenerate(self):
		musicGr = QAction('Music Generate',self)
		musicGr.setShortcut('Ctrl+M')
		musicGr.triggered.connect(self.music_generate)
		return musicGr

	def music_generate(self):
		syll_model_path = './enc_models/syllEncoding_20190419.bin'
		word_model_path = './enc_models/wordLevelEncoder_20190419.bin'
		syllModel = Word2Vec.load(syll_model_path)
		wordModel = Word2Vec.load(word_model_path)

		'''
		lyrics = [['Must','Must'],['have','have'],['been','been'],['love','love'],
				['but','but'],['its','its'],['o','over'],['ver','over'],['now','now'],['lay','lay'],['a','a'],
				['whis','whisper'],['per','whisper'],['on','on'],['my','my'],['pil','pillow'],['low','pillow']]
		lyrics = [['Then','Then'],['the','the'],['rain','rainstorm'],['storm','rainstorm'],['came','came'],
				['ov','over'],['er','over'],['me','me'],['and','and'],['i','i'],['felt','felt'],['my','my'],
				['spi','spirit'],['rit','spirit'],['break','break']]
		lyrics = [['E','Everywhere'],['very','Everywhere'],['where','Everywhere'],['I','I'],['look','look'],
				['I','I'],['found','found'],['you','you'],['look','looking'],['king','looking'],['back','back']]
		'''
		lyrics = [['Must','Must'],['have','have'],['been','been'],['love','love'],
				['but','but'],['its','its'],['o','over'],['ver','over'],['now','now'],['lay','lay'],['a','a'],
				['whis','whisper'],['per','whisper'],['on','on'],['my','my'],['pil','pillow'],['low','pillow'],['Then','Then'],['the','the'],['rain','rainstorm'],['storm','rainstorm'],['came','came'],
				['ov','over'],['er','over'],['me','me'],['and','and'],['i','i'],['felt','felt'],['my','my'],
				['spi','spirit'],['rit','spirit'],['break','break'],['E','Everywhere'],['very','Everywhere'],['where','Everywhere'],['I','I'],['look','look'],
				['I','I'],['found','found'],['you','you'],['look','looking'],['king','looking'],['back','back'],['You','You'],['turn','turn'],['my','my'],['nights','nights'],
				['in','into'],['in','into'],['days','days'],['Lead','Lead'],['me','me'],['mys','mysterious'],['te','mysterious'],
				['ri','mysterious'],['ous','mysterious'],['ways','ways']]
		'''lyrics = [['You','You'],['turn','turn'],['my','my'],['nights','nights'],
				['in','into'],['in','into'],['days','days'],['Lead','Lead'],['me','me'],['mys','mysterious'],['te','mysterious'],
				['ri','mysterious'],['ous','mysterious'],['ways','ways']]'''

		length_song = len(lyrics)
		cond = []

		for i in range(20):
			if i < length_song:
				syll2Vec = syllModel.wv[lyrics[i][0]]
				word2Vec = wordModel.wv[lyrics[i][1]]
				cond.append(np.concatenate((syll2Vec,word2Vec)))
			else:
				cond.append(np.concatenate((syll2Vec,word2Vec)))


		flattened_cond = []
		for x in cond:
			for y in x:
				flattened_cond.append(y)
		
		model_path = './saved_gan_models/saved_model_best_overall_mmd'
# model_path = './saved_gan_models/saved_model_end_of_training'

		x_list = []
		y_list = []

		tf.compat.v1.disable_eager_execution()

		with tf.compat.v1.Session(graph=tf.Graph()) as sess:
			tf.compat.v1.saved_model.loader.load(sess, [], model_path)
			graph = tf.compat.v1.get_default_graph()
			keep_prob = graph.get_tensor_by_name("model/keep_prob:0")
			input_metadata = graph.get_tensor_by_name("model/input_metadata:0")
			input_songdata = graph.get_tensor_by_name("model/input_data:0")
			output_midi = graph.get_tensor_by_name("output_midi:0")
			feed_dict = {}
			feed_dict[keep_prob.name] = 1.0
			condition = []
			feed_dict[input_metadata.name] = condition
			feed_dict[input_songdata.name] = np.random.uniform(size=(1, 20, 3))
			condition.append(np.split(np.asarray(flattened_cond), 20))
			feed_dict[input_metadata.name] = condition
			generated_features = sess.run(output_midi, feed_dict)
			sample = [x[0, :] for x in generated_features]
			sample = midi_statistics.tune_song(utils.discretize(sample))
			midi_pattern = utils.create_midi_pattern_from_discretized_data(sample[0:length_song])
			destination = "test1.mid"
			midi_pattern.write(destination)
		


	def exitAction(self):
		exitAc = QAction('&Exit',self)
		exitAc.setShortcut('Ctrl+Q')
		exitAc.setStatusTip('Exit App')
		exitAc.triggered.connect(self.closeEvent)
		return exitAc

        
	def closeEvent(self,event):
		reply = QMessageBox.question(self,'Message','Pres Yes to Close.',QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)
		
		if reply == QMessageBox.Yes :
			qApp.quit()
		else :
			try:
				event.ignore()
			except AttributeError:
				pass

	def playSound(sound):
		if sound == 0:
			playsound('./Music/emotional-piano-sad-background-music-for-videos-5688-angry.mp3')
		if sound == 1:
			playsound('./Music/easy-country-music-intro-outro-11915-disgusted.mp3')
		if sound == 2:
			playsound('./Music/relaxed-inspiration-ig-version-short-60s-6580-fearful.mp3')
		if sound == 3:
			playsound('./Music/cinematic-fairy-tale-story-short-kikc-8698-happy.mp3')
		if sound == 4:
			playsound('./Music/inspiring-cinematic-uplifting-piano-short-8701-neutral.mp3')
		if sound == 5:
			playsound('./Music/acoustic-guitar-dream-30-seconds-4642-sad.mp3')
		if sound == 6:
			playsound('./Music/FadedPianoCover-VA-4405663-Surprised.mp3')
	

if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = MainWindow()
	sys.exit(app.exec_())
