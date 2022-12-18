You need a color tiff image of 256x256 pixels and 24 bits/pixel. 
Some example images can be downloaded here <https://sipi.usc.edu/database/database.php?volume=misc>. 
I have used the images "4.1.05"-"4.1.07".

To run the project, first install rust and then run this command
```
cargo run --release -- 4.1.05.tiff
```
After the algorithm has converged you can navigate between feature layers with the left and right arrow keys.
You can also change the render resolution with the up and down arrow keys.
Finally if you want to save an image you can use ctrl+s.