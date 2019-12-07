The program demonstrates how simple image processing can be used to track a simple object. 

The problem statement can be found in the Assignment_1_vision.pdf and the approach solution can be found in the Report.pdf file.

## Requirements:
-Python 2.7

-OpenCV

-Numpy

## Code Syntax:
-python test.py input_filename 

-python test1.py input_filename output filename: filename + _out.avi

-python test2.py input_filename output_filename: filename + _2_out.avi

-python test5.py input_filename 
 :output_filename: filename + _5_out.avi

-python test6.py input_filename 
 :output_filename: filename + _6_out.avi

### Different methods have been tried

#### test.py:
just diplays the best line fitting the edges

#### test1.py 
finds the line that is closest to the previous line and displays it

#### test2.py 
finds the line closest to the previous line, then the lines are weighted to as to reduce fluctuations. (Weighted Smoothing)

#### test5.py and above 
finds all lines which start at the top of the image. then takes the lines with most common slope and finds the average among them.
Also does exponential smoothing of the lines so that the lines are not too different from each other, thereby reducing fluctuations.
The only thing different among these are the weights for smoothing and the max number of blank frames.

#### run_all.sh
Basically runs the test6.py on all the videos. To run any other python program on the videos 1-10, change the test6.py in the script with
the other python filename (e.g. test5.py).
