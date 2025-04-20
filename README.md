# 3D_Fractal Generator

## Description
A Python program that renders a 3D version of one of my fractal artworks.

You may choose a value for N, which will determine the size of the image.

This image is composed of cubes attached to cubes at the corners according to some complicated rules.

This is the VisPy + Numba version. The VisPy library makes the image rendering extremely fast. 

The Numba library increases the efficiency of the Numpy calculations (uses decorators to compile functions into machine code format).

This version is my attempt to optimize the code to push the highest values of N possible. 

I successfully computed N=13 on my computer in 3min 39s after 23 color cycles (336,320 cubes placed total).

The main bottleneck issue with this program is that you have extremely large arrays keeping track of cube placement that you must reference for collision checks.

I am not hitting issues with CPU or GPU, it's simply a matter of the RAM having to accommodate these very large arrays. 

My RAM usage hit a peak of 76% during this calculation, while the CPU was only 16%, and the GPU had a tiny 3% spike during the image rendering (Vispy utilizes GPU). 


## Installation
This version is the most complicated in terms of dependencies.

I added a new requirements.txt file to this branch with the general dependencies.

I also added a conda environment file because I used conda to support the numba library.

I was using PyCharm again for this version, and I'll explain how to set it up in this IDE.

First, you'll need to install Anaconda if you don't already have it, at https://www.anaconda.com/download

Then you'll want to create a new project, choose custom environment, generate new, type: conda.

It's best practice to create a new virtual environment specifically for this project to isolate the needed dependencies. 

You'll want to choose view/tool window/terminal (or alt+f12) to open a terminal window if it's not already open at the bottom.

You'll want to activate the conda powershell and activate your virtual environment:

conda init powershell (you may need to restart the terminal after)

conda activate (project name)

Then you can add the environment.yml file I included to your project directory and use this command to install the dependencies:

conda env update -f environment.yml

To verify installation, you can type:

conda list

To check if it matches the dependencies listed in the .yml file. 


## How it works

### Numba version update: 

I removed a filter that would prohibit more than 2 cubes touching at a single vertex to make the numba implementation work better.

This has changed the behavior of the pattern a little where some irregularities can emerge if a cube adds a cube in the available space before another cube gets the chance to do so.

For example, there are 2 2x2x2 cubes next to each other which are equidistant from the origin, but one was added to the list before the other, and it gets to build a 3x3x3 cube in the available space, but this prevents the other cube from doing so.

These irregularities detract from the "perfect symmetry", but they also make the pattern more dynamic, and it was better for the purpose of optimizing processing speed, so I left it that way. 
----update---

This was a rather complex task because the rules of the fractal are complicated.

The rules involve starting with a framework of size N (how many layers to add initially), where cubes of progressively larger size are added to the 8 corners of an initiator cube of size 1x1x1.

A layer will be added of size 2x2x2, then 3x3x3, etc. The framwork only involves expanding along the 8 trajectories extending from the 8 vertices of the initiator cube.

After this framework has been constructed, this will define the outer bounds of the image, and then extra cubes will be added, filling in the available space.

The additional cubes are added in layers. I applied a color coding system to make this more visible and to help with the process of following the rules.

The color code goes: red, orange, yellow, green, blue, then purple (then cycles).

You can only add orange blocks to red blocks, yellow blocks to orange blocks, etc.

The layering process involves cycling through the colors adding only a certain color at a time, then moving on the the next color. 

The rules for adding a cube are: first, identify all cubes of a given color and calculate all their exposed vertices.

Second, organize these vertices in order of which are closest to the origin (center of initiator cube at 0.5, 0.5, 0.5).

Third, attempt to attach a cube to these vertices in this order of size S+1, where S is the length of one of the sides of the cube (parent) you are attaching to.

If a cube of this size does not fit in the available space, then try a cube of size 1x1x1 less, until you can fit a cube, or don't place a cube if even a 1x1x1 cube won't fit.

Cubes are only allowed to be attached at the corners, no touching of faces or edges, and no overlap. 

The layering ends when you have completed a cycle through all colors without being able to place any additional cubes.

I provided reference images of the original artwork, a smaller version with color coding, and some of my results at various values of N.

This version utilizing VisPy is vastly superior to the previous version. I was able to calculate N=10 without a very long processing time at all and my RAM was at about 60%. 

You can probably push higher values using this version, especially if you have good hardware. I provided an image of this N=10 image for reference with a zoomed view as well. 


## Creative process - vibe coding

I'm an aspiring coder, but this was a bit above my level. 

I was vibe-coding in collaboration with Gemini 2.5 Pro Experimental.

I started with a session with default settings for the model, then I moved it to another interface where I could adjust the parameters to be better suited for coding purposes. 

This ended up being like a 20-shot chain of prompts to get this final version.

I was getting stalled with my default settings session, but after adjusting the parameters it was only about 4 shots to get the program where I wanted it.

I used these settings and they seemed pretty effective: Temp: 0.3, Top K: 20, Top P: 0.3.

These values are probably higher than what would be considered optimal, but I wanted to give it freedom for creative solutions since this was an odd project.

I think this fine-tuning might have helped it for my purpose, but perhaps it would have been fine with lower values too. 


## License

I've decided to release this under the The Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.

You are free to duplicate, modify, and share this as long as you credit me and release it under the same license, and do not use it for commercial purposes. 
