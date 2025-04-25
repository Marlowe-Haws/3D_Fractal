# 3D_Fractal Generator Experimental Version

This version is incomplete. I'm experimenting with optimization, and this is just to keep track of my progress.

I was encountering a RAM limit, so I am altering logic for efficiency and may implement CuPy for GPU acceleration. 

--Update 1--

This is now the best version I have, and I'm still improving it. 

I made a silly error checking for orange on red blocks in the first cycle in previous versions; you should start with yellow on orange blocks because that's the first cycle where blocks can be placed. 

I added a pre-filter check to test all candidate vertices to see if you can place a cube of size =1, and if not, to eliminate this from the candidate list. 

This greatly speeds things up by eliminating lots of options quickly and preventing them from being reconsidered in subsequent cycles. 

I've also implemented a grid feature that looks for collisions between the candidate cube and existing cubes by searching the local area, broken into grid-size portions. 

For the most part, breaking things up into 1x1x1 sections is the fastest, but at N=14 and above, it starts to be faster to search 2x2x2 sections. 

It might be faster at higher grid sizes for higher N values, but I haven't tested above n=17 yet, and so far 2 is the highest value that can give a potential calculation speed boost.

So far, this version is using less RAM than previous versions. I remember using Vispy + Numba version and I was hitting 75% RAM usage at N=14 before the VisPy visualization.

Recently, I calculated N=17 using this version and it only reached 62% during the initial calculations, then 76% after VisPy (my background processes account for 37% usage and my total installed RAM is 16GB). 

I've realized that this fractal has become a variant of the original 2D version that inspired it.

In the original, you would build outward from the center in layers.

This version, using the color cycles for values of N=7+ will have a "branching" effect, where cubes may be added close and far from the center if the Framework was large enough to repeat colors.

I don't necessarily think of this as a bad thing, it will just introduce more variability in the size of blocks, because the repeated colors far away from the center will likely have more room for larger cubes.

Rather than it seeming like the primary growth is from the center outward, there will be branches of origins, where growth occurs at different levels simultaneously. 

This variation on the fractal was a side-effect of my goal of maximizing computing speed, and I don't think it really detracts from the beauty of the fractal; it's just a variation. 

--End update 1--

--Update 2--

I've uploaded a working CUDA version. It is not optimized yet, but it functions. 

I'm working on reducing the overhead time of converting CPU batches of information to GPU batches.

I've added an input for size of batches, and total processing time stamps to help with testing. 

I'm hoping this approach will eventually pay off and exceed all my previous versions. 


## License

I've decided to release this under the The Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.

You are free to duplicate, modify, and share this as long as you credit me and release it under the same license, and do not use it for commercial purposes. 
