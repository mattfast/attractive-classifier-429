Datasets from:
  http://www.cc.gatech.edu/~hays/compvision2015/proj5/

training_faces:
  These are 6000 cropped 36x36 faces from the Caltech Web Faces dataset.
  http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/

  They were prepared by James Hays by eliminating images that are not
  large or frontal enough, then cropping and resizing.

training_nonfaces:
  These are 250 non-face images from datasets by Wu et al. and the SUN
  scene database.
  http://c2inet.sce.ntu.edu.sg/Jianxin/RareEvent/rare_event.htm
  http://groups.csail.mit.edu/vision/SUN/

  They were prepared by James Hays by eliminating animal faces, illustrated
  faces, and duplicate images.

testing_faces:
testing_nonfaces:
  These are 500 additional faces and 500 additional 36x36 nonface images.

single_scale_scenes:
  These are 83 images from the CMU/MIT frontal face image database.
  http://vasc.ri.cmu.edu/idb/html/face/frontal_images/

  They were prepared by Szymon Rusinkiewicz by eliminating images with no
  faces, images with faces smaller than 36x36, or images with faces at multiple
  scales.  Then they were rescaled such that the average face size is 36x36.

