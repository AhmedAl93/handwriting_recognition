The script is used to recognize handwritten dates in an image. 

To recognize handwritten dates, run:

python recognizer.py


The algorithm is:
- (1) specify or detect handwritten text areas (to build detection model, check 'detection_model' folder), cut this specific area in the image. 
- (2) refine the text areas
- (3) segment each characters in the text areas: rough segmentation and fine segmenation
- (4) cut and clean image segmenations of each characters
- (5) use a CNN model to recognize each character
- (6) combine segments back to get the top-k recognition results

