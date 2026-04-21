import numpy as np
import pyradi.ryfiles as ryfiles


nfr,img = ryfiles.readRawFrames('data/E-W-m2-detector-100-256.double', 
                        rows=100, 
                        cols=256,
                        vartype=np.float64, loadFrames=[0])

                        
ryfiles.writeRawFrames('data/StairIR[100,256].double', 13.038*img, 'double')
