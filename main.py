import numpy as np
import matplotlib.pyplot as plt
import random, time, json
from colour import *


def tolist(colr):
  return [colr.r, colr.g, colr.b]
  
def gradient(start, end, steps):
  grad = list(start.range_to(end, steps))
  ret = []
  for color in grad:
    ret.append(tolist(color))
  return ret

debug = False
genType = input('Generation type: ')

ujson = True if input('Load Config?: ') in ["yes","y","ok", "sure", "Y", "Yes", "Ok","yee", '1'] else False
if ujson:
    dir = input('Config name: ')
    with open(dir, 'r') as r:
        config = json.load(r)
        csize = config['size']
        cseed = config['seed']
        cwaterlevel = config['waterlevel']
        cnoiseScale = config['noiseScale']
        try:
            cBrownian = config['Brownian']
        except KeyError:
            cBrownian = True
        try:
            cns = config['ns']
        except KeyError:
            if noiseScale:
                cns = int(size*0.0078125)
        tname = input('Map name: ')
else:
    csize = int(input('Terrain size: '))
    cseed = input('Integer terrain seed (blank for random): ')
    cseed = int(cseed) if cseed else random.randint(0, 10000)
    cBrownian = True if input('Use Brownian noise: ') in ["yes","y","ok", "sure", "Y", "Yes", "Ok","yee", '1'] else False
    try:
        cwaterlevel = float(input('Water level (from 0 to 100): '))/100
    except TypeError:
        cwaterlevel = 0.5
    cnoiseScale = cnoiseScale = True if input('Scale Noise?: ') in ["yes","y","ok", "sure", "Y", "Yes", "Ok","yee", '1'] else False
    if not cnoiseScale:
        cns = int(input('Noise Resolution: '))
    else:
        cns = int(csize*0.0078125)
    if input('Save config?: ') in ["yes","y","ok", "sure", "Y", "Yes", "Ok","yee", '1']:
        with open(input('Config name: '), 'w') as s:
            cfg = {
                'size': csize,
                'seed': cseed,
                'waterlevel': cwaterlevel,
                'noiseScale': cnoiseScale,
                'ns': cns,
                'Brownian': cBrownian
            }
            json.dump(cfg, s)
    tname = input('Map Name: ')
np.random.seed(seed=cseed)

cfgMap = {
    'size': csize,
    'seed': cseed,
    'waterlevel': cwaterlevel,
    'noiseScale': cnoiseScale,
    'ns': cns,
    'Brownian': cBrownian
  }


def generate_perlin_noise_2d(shape, res, null):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)
        
def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]), 0)
        frequency *= 2
        amplitude *= persistence
    return noise
    

def e8b(n):
    return int(n*(255))

def rgb(r,g,b):
    return [r,g,b]
  
# Normal Terrain Color Factory  
Ncollist = [[156, 212, 226], [138, 181, 73], [35, 79, 11], [186, 140, 93], [181, 193, 156], [145, 200, 225],  [252, 225, 156], [194, 170, 105], [220,220,220], [190,190,190]]

for x in range(17):
    n = e8b(((x/45)**1.05))+70
    val = [n,n,n]
    Ncollist.append(val)
    if x == 1:
        Ncollistart = Ncollist.index(val)

for x in range(17):
    n = random.choice([(Ncollist[Ncollistart+x-1][0]-20),(Ncollist[Ncollistart+x-1][0]+20), 115])
    val = [n,n,n]
    Ncollist.append(val)
    if x == 1:
        dcollistart = Ncollist.index(val)

grasslist = len(Ncollist)
grass = [rgb(138, 181, 73), rgb(122, 155, 71), rgb(106, 130, 70), rgb(90, 104, 68), rgb(74, 79, 67), rgb(66, 66, 66)]
Ncollist+=grass


Ncolors = np.array(Ncollist ,dtype=np.uint8)
# End Of Normal Color Factory


# Desert Terrain Color Factory
sand = [252, 225, 156]
Desert_collist = [[156, 212, 226], sand, [95, 126, 48], [186, 140, 93]]
dune_gradient = []
dune_start = len(Desert_collist)
Desert_collist.append(dune_gradient)

Dcolors = np.array(Ncollist ,dtype=np.uint8)
# End Of Desert Color Factory


def plotNormal(grid):
    image = Ncolors[grid.reshape(-1)].reshape(grid.shape + (3,))

    plt.imsave(f"{tname}_Map.png", image)
    

def genNormal(cfg):
    # Grid    
    n = cfg['size']
    grid = np.ones((n, n), dtype=np.int32)
    noiseScale = cfg['size']
    Brownian = cfg['Brownian']
    ns = cfg['ns']
    waterlevel = cfg['waterlevel']
    if noiseScale:
        if Brownian:
            noise = generate_fractal_noise_2d((n, n), (int(n*0.0078125), int(n*0.0078125)), 6)
        else:
            noise = generate_perlin_noise_2d((n, n), (int(n*0.0078125), int(n*0.0078125)), 6)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        np.random.seed(seed=cseed+10)
        moisture= generate_perlin_noise_2d((n, n), (int(n*0.0078125), int(n*0.0078125)), 0)
        moisture= (moisture - moisture.min()) / (moisture.max() - moisture.min())
        noise2 = generate_fractal_noise_2d((n, n), (int(n*0.0078125), int(n*0.0078125)), 6)
        noise2 = (noise2 - noise2.min()) / (noise2.max() - noise2.min())
        noise3 = (noise2*2 + (noise)) / 3
    else:
        if Brownian:
            noise = generate_fractal_noise_2d((n, n), (ns, ns), 6)
        else:
            noise = generate_perlin_noise_2d((n, n), (ns, ns), 6)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        np.random.seed(seed=cseed+10)
        moisture= generate_perlin_noise_2d((n, n), (ns, ns), 0)
        moisture= (moisture - moisture.min()) / (moisture.max() - moisture.min())
        noise2 = generate_fractal_noise_2d((n, n), (ns, ns), 6)
        noise2 = (noise2 - noise2.min()) / (noise2.max() - noise2.min())
        noise3 = (noise2*2 + (noise)) / 3
    
    # Mountains Foot
    threshold = 0.73
    for x in range(len(grass)):
        threshold += 0.015
        grid[noise > threshold] = grasslist+x

    # Forests
    # threshold1 = 0.37
    # threshold2 = 0.83
    # threshold = 0.8
    # potential = ((noise3 - threshold) / (1 - threshold))**4 * 0.07
    # mask =  np.logical_and((noise3 > threshold1), np.logical_and((noise < threshold2), (noise > threshold1),)) * (np.random.rand(n, n) < potential)
    # # mask = (grid == 1) * (np.random.rand(n, n) < 0.06)
    # grid[mask] = 2

    threshold1 = 0.4
    threshold2 = 0.83
    potential = ((moisture - threshold + noise) / (1 - threshold))**2.5 * 0.02
    mask =  np.logical_and((noise > threshold1), (noise < threshold2)) * (np.random.rand(n, n) < potential)
    # mask = (grid == 1) * (np.random.rand(n, n) < 0.06)
    grid[mask] = 2

    # Random Trees
    mask = (noise > 0.2) * (np.random.rand(n, n) < 0.001)
    grid[mask] = 2
    
    
    
    # Beach
    threshold = waterlevel+0.05
    grid[noise < threshold] = 6

    # Beach debris
    mask = (noise < waterlevel) * (np.random.rand(n, n) < 0.02)
    grid[mask] = 7

    

    # Water
    threshold = waterlevel
    grid[noise < threshold] = 0
    threshold = waterlevel-0.045
    grid[noise < threshold] = 5
    
    

    

    # Dirt
    mask = (grid == 1) * (np.random.rand(n, n) < 0.01)
    grid[mask] = 3
    
    
    
    
    # Mountains
    # mask = (grid == 2) * (np.random.rand(n,n) < 0.05)
    
    threshold = 0.83
    for x in range(14):
        threshold += 0.01
        if x == 13:
            grid[noise > threshold] = 8
        elif x == 12:
            grid[noise > threshold] = 9
        else:
            grid[noise > threshold] = x+Ncollistart
            mask = (noise > threshold) * (np.random.rand(n, n) < 0.04)
            grid[mask] = x+dcollistart
    threshold = 0.3
    
    
    
    # Plot
    plotNormal(grid)
    

def plotDesert(grid):
    image = Dcolors[grid.reshape(-1)].reshape(grid.shape + (3,))

    plt.imsave(f"{tname}_Map.png", image)
    

def genDesert(cfg):
  # Grid    
	n = cfg['size']
  grid = np.ones((n, n), dtype=np.int32)
  noiseScale = cfg['size']
  Brownian = cfg['Brownian']
  ns = cfg['ns']
  waterlevel = cfg['waterlevel']
	grid = np.ones((n, n), dtype=np.int32)

	# Noise
	if Brownian:
	  noise = generate_fractal_noise_2d((n, n), (ns, ns), 6)
	else:
	  noise = generate_perlin_noise_2d((n, n), (ns, ns))
	noise = (noise - noise.min()) / (noise.max() - noise.min())

	# Water
	threshold = 0.1
	grid[noise < threshold] = 0

	# Shrubs
	mask = (grid == 1) * (np.random.rand(n, n) < 0.02)
	grid[mask] = 3

	# Plot
	plotDesert(grid)


if __name__ == '__main__':
  genType = genType.lower()
  if genType in ['normal', 'n', 'norm', '', ' ', 'standard']:
    genNormal(cfgMap)
  elif genType in ['dry', 'desert', 'barren', 'oasis']:
    genDesert(cfgMap)
  else:
    raise NameError('Generation type not recognized')