import nidaqmx
import numpy as np
import tensorflow as tf
from scipy.optimize import fsolve
from tensorflow.keras.models import load_model


class GMW:

	def __init__(self):

		self.writetask = nidaqmx.Task() #Initiate GMW write & read channels
		self.readtask = nidaqmx.Task()

		self.Va_applied = self.writetask.ao_channels.add_ao_voltage_chan("VectorMagnet/ao0")
		self.Vb_applied = self.writetask.ao_channels.add_ao_voltage_chan("VectorMagnet/ao1")
		self.Vc_applied = self.writetask.ao_channels.add_ao_voltage_chan("VectorMagnet/ao2")

		self.Field_X = self.readtask.ai_channels.add_ai_voltage_chan("VectorMagnet/ai0")
		self.Field_Y = self.readtask.ai_channels.add_ai_voltage_chan("VectorMagnet/ai1")
		self.Field_Z = self.readtask.ai_channels.add_ai_voltage_chan("VectorMagnet/ai2")

		self.std_H = np.array([0.54230507, 0.53158307, 0.16335835])
		self.avg_H = np.array([0.00375433, 0.00437459, -0.00017201])
		self.std_V = np.array([2.69063821, 2.66500921, 2.69274475])
		self.avg_V = np.array([0.0017064, -0.00266611, 0.00056095])

		#config = tf.config.experimental
		#device = config.list_physical_devices('CPU')[0]
		#config.set_memory_growth(device,enable=True)

		print("GMW initialized")


	def M18(self, H):
		model = load_model(r'C:/Users/doc/Desktop/GUI_Harmonic_Measurement/GMW_Learning/models/GMW_model_M18.h5',compile=False)
		H[0,0] = -H[0,0]
		H[0,2] = -H[0,2]
		result = (model.predict((H/2000-self.avg_H)/self.std_H)*self.std_V+self.avg_V)
		if np.max(np.abs(result))>=10:
			return "Your output field is too large"
		else:
			return result[0].tolist()


	def M9(self, H):
		model = load_model(r'C:/Users/doc/Desktop/GUI_Harmonic_Measurement/GMW_Learning/models/GMW_model_M9.h5',compile=False)
		H[0,0] = -H[0,0]
		H[0,2] = -H[0,2]
		result = (model.predict((H/2000-self.avg_H)/self.std_H)*self.std_V+self.avg_V)
		if np.max(np.abs(result))>=10:
			return "Your output field is too large"
		else:
			return result[0].tolist()


	def GMWsolver(self,H):

		def GMWfield(V):

			x,y,z = V[0],V[1],V[2]

			ip = np.array([-4.34e-4,1.05e-4,0.095,-0.014,-7.76,0.457,408.05])
			oop = np.array([0.07,-0.02,-69.57,0,0,0,0])

			pol1=np.array([ip,np.zeros(7),oop])
			pol2=np.array([ip,ip,oop])
			pol3=np.array([ip,ip,oop])

			return [
					(pol1[0,0]*(x**7)+pol1[0,1]*(x**6)+pol1[0,2]*(x**5)+pol1[0,3]*(x**4)+pol1[0,4]*(x**3)+pol1[0,5]*(x**2)+pol1[0,6]*x)*-1+\
					(pol2[0,0]*(y**7)+pol2[0,1]*(y**6)+pol2[0,2]*(y**5)+pol2[0,3]*(y**4)+pol2[0,4]*(y**3)+pol2[0,5]*(y**2)+pol2[0,6]*y)*0.5+\
					(pol3[0,0]*(z**7)+pol3[0,1]*(z**6)+pol3[0,2]*(z**5)+pol3[0,3]*(z**4)+pol3[0,4]*(z**3)+pol3[0,5]*(z**2)+pol3[0,6]*z)*0.5-H[0]/1.35,
					(pol1[1,0]*(x**7)+pol1[1,1]*(x**6)+pol1[1,2]*(x**5)+pol1[1,3]*(x**4)+pol1[1,4]*(x**3)+pol1[1,5]*(x**2)+pol1[1,6]*x)*0+\
					(pol2[1,0]*(y**7)+pol2[1,1]*(y**6)+pol2[1,2]*(y**5)+pol2[1,3]*(y**4)+pol2[1,4]*(y**3)+pol2[1,5]*(y**2)+pol2[1,6]*y)*np.sqrt(3)/2+\
					(pol3[1,0]*(z**7)+pol3[1,1]*(z**6)+pol3[1,2]*(z**5)+pol3[1,3]*(z**4)+pol3[1,4]*(z**3)+pol3[1,5]*(z**2)+pol3[1,6]*z)*-np.sqrt(3)/2-H[1]/1.35,
					pol1[2,0]*(x**3)+pol1[2,1]*(x**2)+pol1[2,2]*x+pol2[2,0]*(y**3)+pol2[2,1]*(y**2)+pol2[2,2]*y+pol3[2,0]*(z**3)+pol3[2,1]*(z**2)+pol3[2,2]*z-H[2]
					]

		result = fsolve(GMWfield,[0,0,0],full_output=True)
		flag = result[2]
	

		if np.max(np.abs(result[0]))>=10:
			return "Your output field is too large"
		elif flag !=1:
			return "The solver fails QQ"
		else:
			return result[0].tolist()


    #Construct a list for different types of angle scan
	def anglescan(self, field, points=120, angi=0, angf=360, scantype='xy'):

		angstep=(angf-angi)/points
		ang=np.arange(angi,angf+angstep,angstep)/180*np.pi
		angmagV=[]
		index=0

		if scantype == 'xy':
			if field <= 300:
				while (len(angmagV)==index) & (len(angmagV)<points+1) :
					H = field*np.array([np.cos(ang[index]), np.sin(ang[index]), 0])
					sol = self.GMWsolver(H)
					if isinstance(sol,str)==False:
						angmagV.append(sol)
					index+=1
			if field > 300:
				while (len(angmagV)==index) & (len(angmagV)<points+1) :
					H = field*np.array([[np.cos(ang[index]), np.sin(ang[index]), 0]])
					sol = self.M18(H)
					if isinstance(sol,str)==False:
						angmagV.append(sol)
					index+=1

		elif scantype == 'xz':
			if field > 1000:
				print('The field may not be accurate')
			while (len(angmagV)==index) & (len(angmagV)<points+1) :
				H = field*np.array([[np.sin(ang[index]), 0, np.cos(ang[index])]])
				sol = self.M9(H)
				if isinstance(sol,str)==False:
					angmagV.append(sol)
				index+=1

		elif scantype == 'yz':
			if field > 1000:
				print('The field may not be accurate')
			while (len(angmagV)==index) & (len(angmagV)<points+1) :
				H = field*np.array([[0, np.sin(ang[index]), np.cos(ang[index])]])
				sol = self.M9(H)
				if isinstance(sol,str)==False:
					angmagV.append(sol)
				index+=1

		if len(angmagV) < points+1:
			print(f'Anglescan fails at {ang[len(angmagV)]} deg')
			return "Anglescan fails QAQ"
		else:
			print('Now running %s anglescan' %(scantype))
			print('Anglescan max output current = %.2g A' %(10*np.max(angmagV)))
			return ang, angmagV


    #Construct a list for low-high field scan with deep learning models
    #20211006 Utilizing deep learning models M9 and M18 to predict voltages
	def fieldscan(self, field, points, phi, theta):

		mag=[]
		magV=[]

		p=phi*np.pi/180
		t=theta*np.pi/180
		unit = np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])

		# Create field magnitude list for low-high scan
		flist=np.arange(field*-1,field+field/points,field/points)
		r_flist=flist[::-1]    #reverse
		mag=flist.tolist()+r_flist.tolist()[1:]
		mag_arr = np.array(mag)
		index = 0


		while (len(magV)==index) & (len(magV)<4*points+1) :
			H = np.array([unit*mag_arr[index]])
			if theta==90 or theta==270:
				sol = self.M18(H)
			else:
				sol = self.M9(H)
			if isinstance(sol,str)==False:
				magV.append(sol)
			index+=1

		if len(magV) < 4*points+1:
			print(f'Fieldscan fails at {mag[len(magV)]} Oe')
			return "Fieldscan fails QAQ"
		else:
			print('Now running field scan at phi %f deg and theta %f deg' %(phi,theta))
			print('Fieldscan max output current = %.2g A' %(10*np.max(magV)))
			return mag, magV

	def StandardFieldscan(self, field, points, phi, theta):

		mag = []
		magV = []

		p = phi*np.pi/180
		t = theta*np.pi/180
		unit = np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])

		# Create field magnitude list for standard scan
		mag = np.arange(field*-1,field+field/points,field/points)
		mag_arr = np.array(mag)
		index = 0

		while (len(magV)==index) & (len(magV)<2*points+1) :
			H = np.array([unit*mag_arr[index]])
			if theta==90 or theta==270:
				sol = self.M18(H)
			else:
				sol = self.M9(H)
			if isinstance(sol,str)==False:
				magV.append(sol)
			index+=1

		if len(magV) < 2*points+1:
			print(f'Fieldscan fails at {mag[len(magV)]} Oe')
			return "Fieldscan fails QAQ"
		else:
			print('Now running field scan at phi %f deg and theta %f deg' %(phi,theta))
			print('Fieldscan max output current = %.2g A' %(10*np.max(magV)))
			return mag, magV

	#Construct a list for standard field scan with an additional Hz (typically for PMA first harmonic)
	def HzBiasedFieldscan(self, field, points, phi, Hz):

		mag = []
		magV = []
		p = phi*np.pi/180


		# Create field magnitude list for standard scan
		mag=np.arange(field*-1,field+field/points,field/points)
		mag_arr = np.array(mag)
		index = 0

		while (len(magV)==index) & (len(magV)<2*points+1) :
			H = np.array([[mag_arr[index]*np.cos(p),mag_arr[index]*np.sin(p),Hz]])
			sol = self.M9(H)

			if isinstance(sol,str)==False:
				magV.append(sol)
			index+=1

		if len(magV) < 2*points+1:
			print(f'Fieldscan fails at {mag[len(magV)]} Oe')
			return "Fieldscan fails QAQ"
		else:
			print('Now running Hz-biased field scan at phi %f deg and Hz %f Oe' %(phi,Hz))
			print('Fieldscan max output current = %.2g A' %(10*np.max(magV)))
			return mag, magV


	#Construct a list for low-high field scan with simple model solver
	def SolverFieldscan(self, field, points, phi, theta):

		mag=[]
		magV=[]

		p  =phi*np.pi/180
		t=theta*np.pi/180
		unit = np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])

		# Create field magnitude list for low-high scan
		flist=np.arange(field*-1,field+field/points,field/points)
		r_flist=flist[::-1]    #reverse
		mag=flist.tolist()+r_flist.tolist()[1:]
		mag_arr = np.array(mag)
		index = 0


		while (len(magV)==index) & (len(magV)<4*points+1) :
			H = unit*mag_arr[index]
			sol = self.GMWsolver(H)
			if isinstance(sol,str)==False:
				magV.append(sol)
			index+=1

		if len(magV) < 4*points+1:
			print(f'Fieldscan fails at {mag[len(magV)]} Oe')
			return "Fieldscan fails QAQ"
		else:
			print('Now running field scan at theta %f deg and phi %f deg' %(phi,theta))
			print('Fieldscan max output current = %.2g A' %(10*np.max(magV)))
			return mag, magV

	#Construct a list for field-scan loopshift measurement
	#20211006 Utilizing deep learning models M9 and M18 to predict voltages
	def loopshift(self, Hx, Hz, points):

		mag=[]
		magV=[]

		# Create field magnitude list for low-high scan
		zlist=np.arange(Hz*-1,Hz+Hz/points,Hz/points)
		r_zlist=zlist[::-1]    #reverse
		mag=zlist.tolist()+r_zlist.tolist()[1:]
		index=0


		while (len(magV)==index) & (len(magV)<4*points+1) :
			H = np.array([[Hx, 0, mag[index]]])
			sol = self.M9(H)
			if isinstance(sol,str)==False:
				magV.append(sol)
			index+=1

		if len(magV) < 4*points+1:
			print(f'Loopshift fails at Hz = {mag[len(magV)]} Oe')
			return "Loopshift fails QAQ"
		else:
			print('Now running Loop shift measurement at Hx %f Oe' %Hx)
			print('Loopshift max output current = %.2g A' %(10*np.max(magV)))
			return mag, magV


	#Construct a list for angle-scan loopshift measurement
	def loopshift_angle(self, Hin, Hz, points, ang):

		mag=[]
		magV=[]

		# Create field magnitude list for low-high scan
		zlist=np.arange(Hz*-1,Hz+Hz/points,Hz/points)
		r_zlist=zlist[::-1]    #reverse
		mag=zlist.tolist()+r_zlist.tolist()[1:]

		index=0

		while (len(magV)==index) & (len(magV)<4*points+1) :
			H = np.array([[Hin*np.cos(ang*np.pi/180), Hin*np.sin(ang*np.pi/180), mag[index]]])
			sol = self.M9(H)
			if isinstance(sol,str)==False:
				magV.append(sol)
			index+=1

		if len(magV) < 4*points+1:
			print(f'Loopshift fails at Hz = {mag[len(magV)]} Oe')
			return "Loopshift fails QAQ"
		else:
			print('Now running Loop shift measurement at Hin %f Oe, angle %f deg' %(Hin,ang))
			print('Loopshift max output current = %.2g A' %(10*np.max(magV)))
			return mag, magV

	#Output an arbitrary field predicted with deep learning models
	def output(self, field, phi, theta):
		Hx = field*np.sin(theta*np.pi/180)*np.cos(phi*np.pi/180)
		Hy = field*np.sin(theta*np.pi/180)*np.sin(phi*np.pi/180)
		Hz = field*np.cos(theta*np.pi/180)
		if theta==90 or theta==270:
			result = self.M18(np.array([[Hx,Hy,Hz]]))
		else:
			result = self.M9(np.array([[Hx,Hy,Hz]]))
		self.write(result)

	#Output an arbitrary field predicted with simple model solver
	def SolverOutput(self, field, phi, theta):
		Hx = field*np.sin(theta*np.pi/180)*np.cos(phi*np.pi/180)
		Hy = field*np.sin(theta*np.pi/180)*np.sin(phi*np.pi/180)
		Hz = field*np.cos(theta*np.pi/180)

		result = self.GMWsolver(np.array([Hx,Hy,Hz]))
		print(result)
		self.write(result)

	#Output zero field and close the tasks
	def zero(self,status=True):

		if status == True :
			self.writetask.write([0,0,0])
			self.writetask.close()
			self.readtask.close()
		else:
			self.writetask.close()
			self.readtask.close()

		print("GMW closed")

	def write(self, Vset):
		self.writetask.write(Vset)

	def read(self):
		return self.readtask.read()

	#Output artificial AC voltage to eliminate remanence field
	def remanence(self):

		t = np.linspace(0, 15, 3000)
		v = 6*np.sin(2*np.pi*2*t)*0.75**t
	
		for i in v:
			self.write([i,i,i])

		self.zero()
		

