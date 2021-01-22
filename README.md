# masterthesis
MSc Thesis: "Quantitative MR interscanner harmonization using image style transfer" TU Delft, code files

Files included:
  -Classifier of scanners: CNN to identify the original scanner from which each image comer from.
  -Classifier of generated images: After training the classifier with real images. This code can be used to test the classification of CycleGAN generated images
  -Registration code
  -CycleGAN with SSIM loss
  -CycleGAN with extra discriminator 
  -Validation code for generated images: histograms, correlations, SSIM, NRMSE...
