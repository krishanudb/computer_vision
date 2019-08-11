## Requirements:
-Python 2.7

-OpenCV

-Numpy

## Code Syntax:
-python test.py input_filename 

-python test1.py input_filename output filename: filename + _out.avi

-python test2.py input_filename output_filename: filename + _2_out.avi

#### The two methods are slightly different

test.py: just diplays the best line fitting the edges

test1.py finds the line that is closest to the previous line and displays it

test2.py finds the line closest to the previous line, then the lines are weighted to as to reduce fluctuations. (Weighted Smoothing)
