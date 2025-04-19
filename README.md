# 3D_Fractal
A program that renders a 3D version of one of my fractal artworks.
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
I was able to calculate N=6, but while N=7 seemed like I was not in danger of crashing my system (my 16GB RAM was steadily at around 50%), I waited more than 30 minutes and it still hadn't rendered the image, so I stopped.
You can calculate higher values at your discretion if you think your computer can handle it, or to stress test your system. 
I'm an aspiring coder, but this was a bit above my level. 
I was vibe-coding in collaboration with Gemini 2.5 Pro Experimental.
I started with a session with default settings for the model, then I moved it to another interface where I could adjust the parameters to be better suited for coding purposes. 
This ended up being like a 20-shot chain of prompts to get this final version.
I was getting stalled with my default settings session, but after adjusting the parameters it was only about 4 shots to get the program where I wanted it.
I used these settings and they seemed pretty effective: Temp: 0.3, Top K: 20, Top P: 0.3.
These values are probably higher than what would be considered optimal, but I wanted to give it freedom for creative solutions since this was an odd project.
I think this fine-tuning might have helped it for my purpose, but perhaps it would have been fine with lower values too. 
