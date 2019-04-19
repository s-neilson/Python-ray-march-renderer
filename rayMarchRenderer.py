import copy
import math
import numpy
from tqdm import tqdm
import matplotlib.pyplot as plt


#A class for three-dimentional vector operations; using numpy arrays and their operations was found to be slower.
class Vector3D():
    x=0.0
    y=0.0
    z=0.0
    
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z
        
    def __add__(self,b):
        return Vector3D(self.x+b.x,self.y+b.y,self.z+b.z)
    
    def __sub__(self,b):
        return Vector3D(self.x-b.x,self.y-b.y,self.z-b.z) 
    
    def __mul__(self,b):
        if((type(b)==float) or (type(b)==int)):
            return Vector3D(self.x*b,self.y*b,self.z*b) #Scalar multiplication
        else:
            return Vector3D(self.x*b.x,self.y*b.y,self.z*b.z) #Element wise multiplication 
            
    def __truediv__(self,b):
        if((type(b)==float) or (type(b)==int)):
            return Vector3D(self.x/b,self.y/b,self.z/b) #Scalar division
        else:
            Vector3D(self.x/b.x,self.y/b.y,self.z/b.z) #Element wise division 

    def dotProduct(self,b):
        return (self.x*b.x)+(self.y*b.y)+(self.z*b.z)
    
    def crossProduct(self,b):
        newX=(self.y*b.z)-(self.z*b.y)
        newY=(self.z*b.x)-(self.x*b.z)
        newZ=(self.x*b.y)-(self.y*b.x)
        
        return Vector3D(newX,newY,newZ)
    
    def lengthSquared(self):
        return (self.x**2)+(self.y**2)+(self.z**2)
    
    def length(self):
        return math.sqrt(self.lengthSquared())
                         
    def getUnitVector(self):
        return self/self.length()
    
    
    
#Returns the unit vector pointing from the location represented by fromVector to the location represented by toVector.
def getUnitVectorPointToPoint(toVector,fromVector):
    differenceVector=toVector-fromVector
    return differenceVector.getUnitVector()

#Returns the direction that the original ray used in rayMarch should be in in order to correspond to a certain pixel in the image given by the coordinates
#pixelX,pixelY. The camera is assumed to be a pinhole comera with an infinitely small aperture.
def getCameraRayUnitVector(pixelX,pixelY,imageWidth,imageHeight,cameraLocation,aspectRatio,screenDistance):
    screenZ=screenDistance+cameraLocation.z
    screenX=cameraLocation.x+numpy.interp(x=pixelX,xp=[0,imageWidth],fp=[-0.5*aspectRatio,0.5*aspectRatio]) #Maps pixel x ordinate to screen x ordinate.
    screenY=cameraLocation.y+numpy.interp(x=pixelY,xp=[0,imageHeight],fp=[-0.5,0.5]) #Maps pixel y ordinate to screen y ordinate.
    
    pixelLocation=Vector3D(float(screenX),float(screenY),float(screenZ))  
    return getUnitVectorPointToPoint(pixelLocation,cameraLocation)



class SceneObject():
    location=0.0
    colour=None
    diffuse=True
    refractiveIndex=1.0
    brightness=0.0
    
    def getNormalUnitVector(self,collisionPoint,rayIsInside):
        return Vector3D(0.0,1.0,0.0)
    
    def getReflectionVector(self,incomingUnitVector,normalUnitVector):
        return incomingUnitVector-((normalUnitVector*(normalUnitVector.dotProduct(incomingUnitVector)))*2.0)
    
    #Returns the angle that the incoming ray makes to the normal using the cosine rule.
    def getIncomingAngle(self,negativeIncomingUnitVector,normalUnitVector):
        negativeIncomingNormalDifferenceVector=negativeIncomingUnitVector-normalUnitVector
        cosIncomingAngle=1.0-(0.5*negativeIncomingNormalDifferenceVector.lengthSquared())
        return math.acos(cosIncomingAngle)
    
    def isAboveCriticalAngle(self,incomingAngle,n1,n2): #Determines whether total internal reflection is occuring.
        if(n2>n1):
            return False #Total internal reflection only occurs when light is in a higher refractive index material and collides with an interface seperating a lower refractive index material.
        
        return incomingAngle>=math.asin(n2/n1)   
    
    #Returns the refractive indices for both sides of the interface during a refractive process
    def getRefractiveIndices(self,rayIsInside):
        #It is assumed that refraction only occurs between this object and empty space (refractive index of 1.0), not between this 
        #object and another object embedded inside it.
        n1=1.0 if(rayIsInside==False) else self.refractiveIndex
        n2=self.refractiveIndex if(rayIsInside==False) else 1.0
        return n1,n2
    
    #Returns the reflection and refraction coefficients using Shlick's approximation (https://en.wikipedia.org/wiki/Schlick%27s_approximation) of the Frensel equations.
    def getReflectionAndRefractionCoefficients(self,incomingAngle,n1,n2):
        verticalReflectionIntensityFactor=((n1-n2)/(n1+n2))**2.0 #Reflection intensity for a ray of light travelling in the negative normal direction.
        reflectionIntensityFactor=verticalReflectionIntensityFactor+((1-verticalReflectionIntensityFactor)*((1-math.cos(incomingAngle))**5.0))
        return reflectionIntensityFactor,1.0-reflectionIntensityFactor
    
    #Returns the information regarding reflection and refraction that occurs when rays of light transition between an interface between two refractive indices.
    #Returns information in the following format: reflection vector, refraction vector,reflection intensity,refraction intensity.
    def getReflectionAndRefraction(self,rayOrigin,collisionPoint,rayIsInside):
        incomingUnitVector=getUnitVectorPointToPoint(collisionPoint,rayOrigin)
        negativeIncomingUnitVector=incomingUnitVector*(-1.0)
        normalUnitVector=self.getNormalUnitVector(collisionPoint,rayIsInside)
        
        reflectionVector=self.getReflectionVector(incomingUnitVector=incomingUnitVector,normalUnitVector=normalUnitVector)
        incomingAngle=self.getIncomingAngle(negativeIncomingUnitVector=negativeIncomingUnitVector,normalUnitVector=normalUnitVector)
        
        n1,n2=self.getRefractiveIndices(rayIsInside=rayIsInside)
        if(self.isAboveCriticalAngle(incomingAngle=incomingAngle,n1=n1,n2=n2)):
            return reflectionVector,Vector3D(0.0,0.0,0.0),1.0,0.0 #No refraction occurs in this case; only total internal reflection. 
        
        negativeNormalUnitVector=normalUnitVector*(-1.0) #Used in construction of the final refracted vector
        parallelVector=(negativeIncomingUnitVector.crossProduct(normalUnitVector)).crossProduct(normalUnitVector)
        parallelUnitVector=parallelVector.getUnitVector() #Parallel to the surface of the object, used in construction of the final refracted vector.
        
        sinRefractionAngle=(n1/n2)*math.sin(incomingAngle) #Calculated from Snell's law.
        refractionAngle=math.asin(sinRefractionAngle)
        
        #Below constructs the refraction vector from negativeNormalUnitVector and parallelUnitVector.
        parallelUnitVectorFactor=math.tan(refractionAngle) #Assumes that the component of the refraction vector in the negative normal direction has a length of 1.
        refractionVector=negativeNormalUnitVector+(parallelUnitVector*parallelUnitVectorFactor)
        unitRefractionVector=refractionVector.getUnitVector()
        
        reflectionIntensityFactor,refractionIntensityFactor=self.getReflectionAndRefractionCoefficients(incomingAngle=incomingAngle,n1=n1,n2=n2)   
        return reflectionVector,unitRefractionVector,reflectionIntensityFactor,refractionIntensityFactor 
        
    
    def isLight(self): #Returns if the brightness is positive, meaning that the object is a light.
        return False if(self.brightness<=0.0) else True    

#An infinite flat plane described by the equation ax+by+cz+d=0
class Plane(SceneObject):    
    normalUnitVector=None
    d=0.0
    
    def __init__(self,a,b,c,location,colour,diffuse,refractiveIndex,brightness):
        self.normalUnitVector=Vector3D(a,b,c).getUnitVector() #From the point-normal definition of a plane
        self.a=a
        self.b=b
        self.c=c
        self.location=location
        self.d=(-1.0)*((self.normalUnitVector).dotProduct(location)) #From the point-normal definition of a plane.
        self.colour=colour.getUnitVector()
        self.diffuse=diffuse
        self.refractiveIndex=refractiveIndex
        self.brightness=brightness
        
    def getNormalUnitVector(self,collisionPoint,rayIsInside):
        return self.normalUnitVector if(rayIsInside==False) else (self.normalUnitVector)*(-1.0) #The normal direction is swapped depending what side of the plane collisionPoint is on.
        
        
    def SDF(self,collisionPoint):
        return ((self.normalUnitVector).dotProduct(collisionPoint))+self.d #Perpendicular distance from point to plane (assuming that the normal vector is a unit vector) considering what side the point is to the plane.
    
    
    def getBrightness(self,point):
        return max(self.brightness,0.0) #Brightness for an infinite illuminated plane is always constant irrespective of the viewing location.
    
    
    
class Sphere(SceneObject):
    radius=1.0
    
    def __init__(self,location,radius,colour,diffuse,refractiveIndex,brightness):
        self.location=location
        self.radius=radius
        self.colour=colour.getUnitVector()
        self.diffuse=diffuse
        self.refractiveIndex=refractiveIndex
        self.brightness=brightness
        
    def getNormalUnitVector(self,collisionPoint,rayIsInside):
        normalUnitVector=getUnitVectorPointToPoint(collisionPoint,self.location) #Normal vectors for a sphere point from the centre to surface points.
        return normalUnitVector if(rayIsInside==False) else (normalUnitVector)*(-1.0) #The normal direction is swapped depending if collisionPoint is internal or external to the sphere.
             
    def SDF(self,collisionPoint):
        pointToSphereCentreVector=self.location-collisionPoint
        distanceToSphereCentre=pointToSphereCentreVector.length()
        return distanceToSphereCentre-self.radius #Gives the distance from collisionPoint to the sphere's surface, is negative if collisionPoint is inside the sphere.
    
    def getBrightness(self,point):
        if(self.brightness<=0.0):
            return 0.0
        
        pointToSphereCentreVector=self.location-point
        distanceSquaredToSphereCentre=pointToSphereCentreVector.lengthSquared()
        return (self.brightness*(self.radius**2.0))/distanceSquaredToSphereCentre #Spherical lights use an inverse square relation for light intensity. Intensity is equal to the value of "brightness" at light surface. 
 
    

    
#Determines the total intensity of red, green and blue light impacting point rayOrigin using the ray marching algorithm. 
def marchRay(currentRecursiveDepth,objectList,originObject,rayOrigin,rayDirection,
             minimumCollisionDistance,maximumRayMarchStepCount,maximumRayMarchDistance,
             maximumRecursiveDepth,minimiseDiffuseInterreflections):

    if(currentRecursiveDepth>=maximumRecursiveDepth):
        return Vector3D(0.0,0.0,0.0) #The path to the camera via recursive calls of marchRay has become to long, no light is returned in this case.

    currentRayEnd=copy.deepcopy(rayOrigin) #Holds the current endpoint of the extended ray that begins at the point rayOrigin.
    currentStepCount=0
    
 
    while(True): 
        if((currentStepCount>=maximumRayMarchStepCount) or (currentRayEnd.length()>=maximumRayMarchDistance)):
            return Vector3D(0.0,0.0,0.0) #The ray has not intersected with anything within a specified number of steps or ray extension distance; the ray may have gone outside of the scene. No light is returned in this case.
        
        closestObject=None
        closestObjectDistance=10e12 #Holds the distance to the closest object
        closestObjectRayIsInside=False #Holds whether the ray is would be colliding with closestObject from inside the object or not.
        
        for currentObject in objectList:
            if(currentObject==originObject): #If the current object being considered is the object that this call of rayMarch is being used on. Used during diffuse reflections.
                continue
            
            currentObjectDistance=currentObject.SDF(currentRayEnd) #The closest distance to the object is determined, with the sign determining what side the collision point is to the surface normals.
            currentObjectAbsoluteDistance=math.fabs(currentObjectDistance)
                     
            if(currentObjectAbsoluteDistance<closestObjectDistance): #If a new closest object has been found to the ray end.
                closestObject=currentObject
                closestObjectDistance=currentObjectAbsoluteDistance
                closestObjectRayIsInside=False if(currentObjectDistance>=0.0) else True
        
        if(closestObjectDistance<=minimumCollisionDistance): #If the ray has collided with the closest object.
            if(closestObject.isLight()==True): #If the object is a light, its colour scaled by its brightness at the ray origin is returned.
                return closestObject.colour*closestObject.getBrightness(rayOrigin)
            
            #In some circumstances multiple objects may have collisions occuring at currentRayEnd at the same time. If this happens, a ray sent from one of these objects another
            #will need no iterations to collide, causing rayOrigin to equal currentRayEnd, meaning that the ray from one to the other will have zero length. To prevent this,
            #the collision point is offset backwards along the incoming by a distance greater than minimumCollisionDistance in order to ensure that a ray going
            #from this collided object always needs at least one iteration to collide with another object, ensuring that any ray emanating from the point rayOrigin has a non zero length.
            reflectionRayOriginOffsetVector=rayDirection*minimumCollisionDistance*(-1.1)
            reflectionRayOrigin=currentRayEnd+reflectionRayOriginOffsetVector
            
            if(closestObject.diffuse==False): #If the object reflects and refracts light.
                #If the ray is transitioning across the object's surface (either from inside to outside or outside to inside) the ray end is moved across the collision interface
                #so that the new ray will not initally collide with the object that it was generated from.
                closestObjectCollisionNormalUnitVector=closestObject.getNormalUnitVector(collisionPoint=currentRayEnd,rayIsInside=closestObjectRayIsInside)
                refractionRayOriginOffsetVector=closestObjectCollisionNormalUnitVector*minimumCollisionDistance*2.0*(-1.1)
                refractionRayOrigin=currentRayEnd+refractionRayOriginOffsetVector


                reflectionVector,refractionVector,reflectionIntensityFactor,refractionIntensityFactor=closestObject.getReflectionAndRefraction(rayOrigin=rayOrigin,
                                                                                                                                               collisionPoint=currentRayEnd,
                                                                                                                                               rayIsInside=closestObjectRayIsInside)

                
                reflectionIntensity=marchRay(currentRecursiveDepth=currentRecursiveDepth+1,objectList=objectList,originObject=None,
                                           rayOrigin=reflectionRayOrigin,rayDirection=reflectionVector,minimumCollisionDistance=minimumCollisionDistance,
                                           maximumRayMarchStepCount=maximumRayMarchStepCount,maximumRayMarchDistance=maximumRayMarchDistance,
                                           maximumRecursiveDepth=maximumRecursiveDepth,minimiseDiffuseInterreflections=minimiseDiffuseInterreflections)
                
                refractionIntensity=marchRay(currentRecursiveDepth=currentRecursiveDepth+1,objectList=objectList,originObject=None,
                                           rayOrigin=refractionRayOrigin,rayDirection=refractionVector,minimumCollisionDistance=minimumCollisionDistance,
                                           maximumRayMarchStepCount=maximumRayMarchStepCount,maximumRayMarchDistance=maximumRayMarchDistance,
                                           maximumRecursiveDepth=maximumRecursiveDepth,minimiseDiffuseInterreflections=minimiseDiffuseInterreflections)
                
                finalIntensities=(reflectionIntensity*reflectionIntensityFactor)+(refractionIntensity*refractionIntensityFactor)
                return closestObject.colour*finalIntensities #The object's surface can reflect and refract red, green and blue light in different amounts.
                
            
            #The light reflects diffusely for the object.
            totalReflectedIntensity=Vector3D(0.0,0.0,0.0) #Carries the current sum of light intensity to be reflected off the object.
            for currentObject in objectList:
                if(currentObject==closestObject): #If the current object being considered is the object that this part of the function is being used on.
                    continue
                
                if((minimiseDiffuseInterreflections==True) and (currentObject.isLight()==False)): #Minimising diffuse interreflection means that diffuse objects will only try to use 
                    #lights and objects in the way of lights as light sources. This simplification can significantly reduce the number of total calculations needed for calculating the colour of a single pixel in the final image.
                    continue
                
                lightDirectionUnitVector=getUnitVectorPointToPoint(currentObject.location,currentRayEnd)
                incomingIntensity=marchRay(currentRecursiveDepth=currentRecursiveDepth+1,objectList=objectList,originObject=closestObject,
                                           rayOrigin=reflectionRayOrigin,rayDirection=lightDirectionUnitVector,minimumCollisionDistance=minimumCollisionDistance,
                                           maximumRayMarchStepCount=maximumRayMarchStepCount,maximumRayMarchDistance=maximumRayMarchDistance,
                                           maximumRecursiveDepth=maximumRecursiveDepth,minimiseDiffuseInterreflections=minimiseDiffuseInterreflections)


                #The Lambertian reflectance model (https://en.wikipedia.org/wiki/Lambertian_reflectance) is used as a model for diffuse reflection.
                surfaceNormalVector=closestObject.getNormalUnitVector(currentRayEnd,False)
                reflectedIntensityScalingFactor=incomingIntensity*max(0.0,lightDirectionUnitVector.dotProduct(surfaceNormalVector))
                reflectedIntensity=closestObject.colour*reflectedIntensityScalingFactor #The object's colour determines how much red, green and blue light is reflected.
                totalReflectedIntensity+=reflectedIntensity #The intensity from the current light source is added to the total sum of reflected light intensity.
                
        
            return totalReflectedIntensity
        #No object intersections exist within a sphere centred at currentRayEnd with a radius of closestObjectDistance. The ray is therefore extended in rayDirection by 
        #a distance slightly smaller than closestObjectDistance (smaller to ensure the ray end will not end up inside an object). 
        currentRayEnd+=(rayDirection*0.98*closestObjectDistance)
        currentStepCount+=1



    
aspectRatio=1.0
imageHeight=50
imageWidth=int(aspectRatio*imageHeight)
fieldOfView=math.pi/2.0
cameraLocation=Vector3D(0.0,0.0,0.0)
screenDistance=(0.5*aspectRatio)/(math.tan(fieldOfView/2.0)) #The screen is assumed to have a height of 1 unit, meaning that it's width is equal to "aspectRatio".
imageData=numpy.zeros(shape=(imageHeight,imageWidth,3))

minimumCollisionDistance=0.005
maximumRayMarchStepCount=400
maximumRayMarchDistance=300.0
maximumRecursiveDepth=7
minimiseDiffuseInterreflections=True
   


#The scene is created below
ground=Plane(a=0.0,b=1.0,c=0.0,location=Vector3D(0.0,-2.0,0.0),colour=Vector3D(255.0,70.0,40.0),diffuse=True,refractiveIndex=1.0,brightness=0.0)
backWall=Plane(a=0.0,b=0.0,c=-1.0,location=Vector3D(0.0,0.0,40.0),colour=Vector3D(170.0,180.0,250.0),diffuse=True,refractiveIndex=1.0,brightness=0.0)
leftWall=Plane(a=1.0,b=0.0,c=0.0,location=Vector3D(-20.0,0.0,-0.0),colour=Vector3D(230.0,240.0,50.0),diffuse=True,refractiveIndex=1.0,brightness=0.0)
rightWall=Plane(a=-1.0,b=0.0,c=0.0,location=Vector3D(20.0,0.0,20.0),colour=Vector3D(230.0,240.0,50.0),diffuse=True,refractiveIndex=1.0,brightness=0.0)
frontWall=Plane(a=0.0,b=0.0,c=1.0,location=Vector3D(0.0,0.0,-40.0),colour=Vector3D(100.0,190.0,170.0),diffuse=True,refractiveIndex=1.0,brightness=0.0)
topLight=Plane(a=0.0,b=-1.0,c=0.0,location=Vector3D(0.0,30.0,0.0),colour=Vector3D(255.0,255.0,255.0),diffuse=True,refractiveIndex=1.0,brightness=1.0)

sphere1=Sphere(location=Vector3D(-2.0,-0.25,5.0),radius=1.5,colour=Vector3D(215.0,250.0,190.0),diffuse=False,refractiveIndex=1.5,brightness=0.0)
sphere2=Sphere(location=Vector3D(0.5,1.0,6.0),radius=1.2,colour=Vector3D(255.0,255.0,255.0),diffuse=False,refractiveIndex=100.0,brightness=0.0)
sphere3=Sphere(location=Vector3D(1.5,0.6,3.0),radius=0.5,colour=Vector3D(80.0,40.0,120.0),diffuse=True,refractiveIndex=1.0,brightness=0.0)
sphereLight1=Sphere(location=Vector3D(15.0,8.0,-5.0),radius=0.01,colour=Vector3D(255.0,255.0,255.0),diffuse=True,refractiveIndex=1.0,brightness=2500000.0) 
   
objectList=[ground,backWall,leftWall,rightWall,frontWall,topLight,sphere1,sphere2,sphere3,sphereLight1]


#These loop through every pixel in the image.
for pixelX in tqdm(range(0,imageWidth)):
    for pixelY in range(0,imageHeight):
            currentPixelColour=Vector3D(0.0,0.0,0.0)
            rayDirection=getCameraRayUnitVector(pixelX=pixelX,pixelY=pixelY,imageWidth=imageWidth,imageHeight=imageHeight,cameraLocation=cameraLocation,
                                                aspectRatio=aspectRatio,screenDistance=screenDistance)
            currentPixelColour+=marchRay(currentRecursiveDepth=0,rayOrigin=cameraLocation,originObject=None,rayDirection=rayDirection,objectList=objectList,
                                         minimumCollisionDistance=minimumCollisionDistance,maximumRayMarchStepCount=maximumRayMarchStepCount,maximumRayMarchDistance=maximumRayMarchDistance,
                                         maximumRecursiveDepth=maximumRecursiveDepth,minimiseDiffuseInterreflections=minimiseDiffuseInterreflections)
        
            imageData[(imageHeight-1)-pixelY,pixelX,:]=[currentPixelColour.x,currentPixelColour.y,currentPixelColour.z] #Y axis is inverted so the image is displayed the correct way up while using imshow().


#The RGB intensity values are scaled from 0 to 1 so they can be interpreted correctly by the imshow() function.
imageDataMaximumValue=numpy.amax(imageData)
imageData/=imageDataMaximumValue
        
plt.imshow(imageData)
plt.show()