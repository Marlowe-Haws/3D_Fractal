# 3D_Fractal
A program that renders a 3D version of one of my fractal artworks.
This was a rather complex task because the rules of the fractal are complicated.
The rules involve starting with a framework, where cubes of progressively larger size are added to the 8 corners of an initiator cube of size 1x1x1.
A layer will be added of size 2x2x2, then 3x3x3, etc. The framwork only involves expanding along the 8 trajectories extending from the 8 vertices of the initiator cube.
After this framework has been constructed, this will define the outer bounds of the image, and then extra cubes will be added, filling in the available space.
The additional cubes are added in layers. I applied a color coding system to make this more visible and to help with the process of following the rules.
The color code goes: red, orange, yellow, green, blue, then purple (then cycles back through).
You can only add orange blocks to red blocks, yellow blocks to orange blocks, etc.
The layering process involves cycling through the colors adding only a certain color at a time, then moving on the the next color. 
The rules for adding a cube are: first, identify all cubes of a given color and calculate all their exposed vertices.
Second, organize these vertices in order of which are closest to the origin (center of initiator cube 0.5, 0.5, 0.5).
Third, attempt to attach a cube to these vertices in this order of size N+1, where N is the length of one of the sides of the cube (parent) you are attaching to.
If a cube of this size does not fit in the available space, then try a cube of size 1x1x1 less, until you can fit a cube, or don't fit a cube if even a 1x1x1 cube won't fit.
