


#code originally by Joe Kington
#https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system

# see also reference images at http://sipi.usc.edu/database/

import numpy as np
import Image

import pyradi.ryplot as ryplot
import pyradi.ryutils as ryutils




if __name__ == '__main__':


    # im = Image.open('lena512color.png')
    im = Image.open('600px-Siemens_star-blurred.png').convert('RGB')
    # im = im.convert('RGB')
    data = np.array(im)

    """Plots an image reprojected into polar coordinages with the origin
    at "origin" (a tuple of (x0, y0), defaults to the center of the image)"""

    print(data.shape)
    origin = None #(300,350)
    pim = ryplot.ProcessImage()
    polar_grid, r, theta = pim.reproject_image_into_polar(data, origin, False)
    p = ryplot.Plotter(1,1,2)
    p.showImage(1, data, ptitle='Image')
    p.showImage(2, polar_grid, ptitle='Image in Polar Coordinates',
        xlabel='Angle',ylabel='Radial')
    p.saveFig('warpedStar.png')
    print('done')