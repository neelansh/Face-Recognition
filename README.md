# Face-Recognition (narendra modi or arvind kejriwal)

## Face Recognition Pipeline

<h4>Step 1: Detect faces</h4>
<p>Faces are detected using current state of the art HOG(Histogram of oriented gradients) using Dlib and opencv for image processing. 
</p>
<h4>Step 2: Align face using facial land marks</h4>
<p>Using Face Landmark Estimation algorithm we are going to detect the Facial featureand the apply tranformations to align the face. It detects 68 landmarks on the face and align the face according to those features.
</p>
<h4>Step 3: Generate Embeddings using NN2 Facenet</h4>
<p>
Generate 128 embeddings as described in the paper facenet(https://arxiv.org/pdf/1503.03832.pdf) using the openface's implementation of facenet.
<strong>Transfer Learning can also be used to extract features from face</strong>
<br>
After applying PCA on the Embeddings generated for the dataset. gives the following result:
<br>
<img src="pca_embeddings_openface.png">
<br>
As we can easily observe the 2 clusters. We can use linear kernel in SVM to classify these images. Also other classifiers where used such as KNN, Random Forest. 
</p>
<h4>Step 4: Train classifiers for Recognition</h4>
<p>
SVM with Linear kernel, SVM with RBF Kernel, KNN, Random forest classifier where trained using SKlearn.
</p>
<hr />
<h2>Data collection</h2>
<p>All data was collected using google images and "Fatkun Batch Download Image" google chrome plugin</p>
<p>Number of Modi images: 135 <br>
Number of Kejriwal images: 121</p>
  
### references

1. https://cmusatyalab.github.io/openface/
2. https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
3. https://arxiv.org/abs/1503.03832
