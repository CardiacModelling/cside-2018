Note that for model 1, the time points are different to what was previously in the code.

Previously, the time points were hard coded such that they covered areas where the signal changed rapidly. However, these timepoints only make sense for the default parameter choice.

There was no way to change the parameters for the competition and keep the same time points (without providing you with a useless dataset).

The dataset contained in this compressed folder contains the correct time points for the competition data (the first column of Cardiac_Data_All.csv or Cardiac_Data_Time.csv).

The new time points have 150 points linearly spaced between 0 and 300 and then 55 points linearly spaced between 301 and 900. This has been updated in the online code.

Note also that the initial conditions for the model are (-10,1,0). Ensure that you use these and NOT the code default settings for initial conditions of (Estar+1, 1, 0).

The SNR for this problem is 10 for each model output (E, h and n).