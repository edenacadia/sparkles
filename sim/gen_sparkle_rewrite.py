## Eden McEwen
#Sparkles rewritten from MagAO-X Code

import hcipy


#m_dmName ## < The descriptive name of this dm. Default is the channel name.
#m_dmChannelName ## < The name of the DM channel to write to.
#m_dmTriggerChannel ## < The DM channel to monitor as a trigger

m_triggerSemaphore  = 9 ## < The semaphore to use (default 9)
m_trigger = True ## < Run in trigger mode if true (default)
m_separation = 15 ##< The radial separation of the speckles (default 15.0)

m_angle = 0.0 ## < The angle of the speckle pattern c.c.w. from up on camsci1/2 (default 0.0)
m_angleOffset = 28.0 ## < The calibration offset of angle so that up on camsci1/2 is 0
m_amp = 0.01 ## < The speckle amplitude on the DM
m_cross = True ## < If true, also apply the cross speckles rotated by 90 degrees
m_frequency = 2000 ## < The frequency to modulate at if not triggering (default 2000 Hz)

mx::improc::eigenCube<realT> m_shapes;
      
IMAGE m_imageStream; 
m_width = 0 ## < The width of the image
m_height = 0 ## < The height of the image.
   
IMAGE m_triggerStream;

m_dataType = 0 ## < The ImageStreamIO type code.
m_typeSize = 0 ## < The size of the type, in bytes.  
   
m_opened = True
m_restart = False
m_modulating = False


def generateSpeckles(m_width, m_height, pixels m_seperation=15, m_angle = 0.0, m_angleOffset=28.0, m_amp=0.01, m_cross=True):
   mx::improc::eigenImage<realT> onesp, onespC; # creating an imaginary field 

   grid = hcipy.make_uniform_grid(pixels, [m_width,m_height])

   onesp.resize(m_width, m_height); # setting height and width of the real field
   onespC.resize(m_width, m_height); # setting height and width of the imaginary field

   m_shapes.resize(m_width, m_height, 4); #???? sets up the four different sparkle shapes
   

   realT m = m_separation * cos( mx::math::dtor<realT>(-1*m_angle + m_angleOffset)); # cosine of angle
   realT n = m_separation * sin( mx::math::dtor<realT>(-1*m_angle + m_angleOffset)); # sine of angle

   mx::sigproc::makeFourierMode(m_shapes.image(0), m, n, 1);
   
   if m_cross: # this means we're setting twice
      onesp = m_shapes.image(0);   
      mx::sigproc::makeFourierMode(m_shapes.image(0), -n, m, 1); # making a FM with a 
      m_shapes.image(0) += onesp;

   m_shapes.image(0) *= m_amp;
   m_shapes.image(1) = -1*m_shapes.image(0);

   mx::sigproc::makeFourierMode(m_shapes.image(2), m, n, -1);

   if m_cross:
      onesp = m_shapes.image(2);
      mx::sigproc::makeFourierMode(m_shapes.image(2), -n, m, -1);
      m_shapes.image(2) += onesp;

   m_shapes.image(2) *= m_amp;
   m_shapes.image(3) = -m_shapes.image(2);

   mx::fits::fitsFile<realT> ff;
   ff.write("/tmp/specks.fits", m_shapes);

   #updateIfChanged(m_indiP_separation, "current", m_separation);
   #updateIfChanged(m_indiP_angle, "current", m_angle);
   #updateIfChanged(m_indiP_amp, "current", m_amp);
   #updateIfChanged(m_indiP_frequency, "current", m_frequency);

   return 0;