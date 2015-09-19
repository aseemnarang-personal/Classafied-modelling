import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage.filters import convolve
import moviepy.editor as mpy

# function to diffuse at next time step

dispersion_kernel = np.array([[0.5, 1 , 0.5],
                                [1  , -6, 1],
                                [0.5, 1, 0.5]])
dispersion_kernel2 = np.array([[0.5, 1 , 0.5],
                                [1  , 0, 1],
                                [0.5, 1, 0.5]])


def dispersion(concentrations_matrix, dispersion_kernel, dispersion_rates, diffusion_zone,
               regions_at_the_border):
    """ Computes the dispersion """
    dispersed = np.array( [convolve(e*diffusion_zone, dispersion_kernel, cval=0)
                           for e in concentrations_matrix]) *diffusion_zone
    correction =  (6-regions_at_the_border) * concentrations_matrix*diffusion_zone
    return dispersion_rates * (dispersed + correction)

def diffuse(diffMat, concMat, diffusion_zone, regions_at_the_border):
    return dispersion(concentrations_matrix=concMat,
                      dispersion_kernel=dispersion_kernel,
                      dispersion_rates=diffMat,
                      diffusion_zone = diffusion_zone,
                      regions_at_the_border= regions_at_the_border)


#function to react according to the concentration in the next time step

decay_rates = np.array([
        0.05, # reactant
        0, # enzyme
        0.05 #product
    ]).reshape((3,1,1))

reaction_rate = 0.05
def react(concMat):
    decay_reaction = decay_rates * concMat
    reactant, enzymes, colorent = concMat
    colorent_producted = reaction_rate * reactant * enzymes
    products_reaction = np.array([
            - colorent_producted,
            0*colorent_producted,
            colorent_producted
        ])

    return  products_reaction - decay_reaction

diffMat= np.array([4, 0, 0.8]).reshape((3,1,1))
dt=0.065

biosensor = mpy.ImageClip("biosensor.png").resize(width=200)
img = 255 * (biosensor.img > 250)
enzymes = img[:,:,2]
reactant = 255*img[:,:,0]
diffusion_zone = 1.0 - (img[:,:,1].astype(float)/255)

im = convolve(diffusion_zone , dispersion_kernel2, cval=0)
regions_at_the_border = im * diffusion_zone

concMat = np.zeros( (3,biosensor.h, biosensor.w),  dtype=float)    #not sure about this
concMat[0] = reactant
concMat[1] = enzymes

world = {'concs':concMat, 't':0}

def update(world):
    world['concs'] += dt * diffuse(diffMat,world['concs'], diffusion_zone, regions_at_the_border)
    world['concs'] += dt * react(world['concs'])
    world['t'] += dt


# ANIMATION 

diffusion_drawing = np.dstack(3*[255*(1-diffusion_zone)])

def world_to_npimage(world):
    #Converts the world's map into a RGB image for the final video.
    coefs = np.array([1,0.05,25]).reshape((3,1,1))
    accentuated_world = world['concs'] * coefs
    image = accentuated_world[::-1].swapaxes(0,2).swapaxes(0,1)
    image += diffusion_drawing
    return np.minimum(255, image)

def make_frame(t):
    #Return the frame for time t
    while world['t'] < t:
        update(world)
    return world_to_npimage(world)

animation = mpy.VideoClip(make_frame, duration=30)
# You can write the result as a gif (veeery slow) or a video:
#animation.write_gif(make_frame, fps=15)
animation.write_gif('model.gif', fps=5)

