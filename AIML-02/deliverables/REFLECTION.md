# Reflection: AIML-02, AI-assisted medical imaging review

**Ramnarayanan M, Nexvia Technologies AI/ML internship**

## What I owned

I owned this project from the data to the review screen. I chose the datasets (NIH ChestX-ray14 for the chest model, Medical Segmentation Decathlon Task 09 for the spleen), trained and evaluated both models, wrote the FastAPI service and the React review screen, and set up the DVC pipeline, the Docker images, the Orthanc connection and the test suite. The decisions on how to split the data, which numbers to report and what to leave out were mine too.

## What I built

A chest X-ray model (DenseNet-121, 14 findings) and a spleen CT model (MONAI 3D U-Net), behind one service and one review screen. A reviewer sees the scores, a Grad-CAM heat map and a SHAP view, then approves, rejects or flags the result. Every action goes into an audit log together with the SHA-256 of the model file that produced it. The whole thing carries a clear "education only, not clinical use" notice.

## What I learned

- **A score is not a product.** The model was the smaller part of the work. The explanation, the record of decisions and the checks around it are what make a result something a person can use.
- **How you split data decides how honest the numbers are.** My first spleen result looked strong, but I had measured it on cases that were also used to choose the model. I retrained on a frozen 29/6/6 split and reported the lower figure, a mean Dice of 0.887 on six held-out cases. I would rather defend a smaller number than explain a bigger one.
- **Raw scores need calibration.** A raw 0.3 is not a 30% chance. Per-label calibration brought the Brier score from 0.090 to 0.053, and I scored it out-of-fold by patient so each image was judged by a calibration fitted on other patients.
- **Negative results are results.** Flipping the CT and averaging made Dice worse, and a learned head on the frozen chest features scored below the model I already had. I dropped both and kept the evidence in the repo.
- **Small samples limit every claim.** The chest numbers come from 999 images and the spleen numbers from six cases. I wrote the limits next to the results instead of hiding them: noisy report-mined labels, about four in five review flags still being false alarms, and lower performance for patients aged 60 and over, which I did not investigate.

## Challenges

The spleen model produced between 43 and 370 stray pieces per case. Keeping only the largest connected blob lifted the mean Dice from 0.715 to 0.877, and a threshold of 0.4, picked on validation cases, took it to 0.887. Getting the Orthanc connection, the audit log and the Docker setup to behave together took the most careful testing, which is why the project has 106 tests and a live Orthanc check.

## What I would do next

Test on data from other hospitals and scanners, add more spleen cases and abnormal anatomy, investigate the gap for older patients, and write results back to the hospital system as DICOM objects.
