# WineQualityPrediction
Wine quality prediction ML model in Spark over AWS


Description: You have to build a wine quality prediction ML model in Spark over AWS. The model must be trained in parallel using 4 EC2 instances. Then, you need to save and load the model in a Spark application that will perform wine quality prediction; this application will run on one EC2 instance.


## Setting up the EMR Cluster

1. In the AWS Learning Lab, search for EMR and open the EMR console
2. Create a new cluster

   a. Names and applications

      i. Choose a name for your cluster

      ii. Choose emr-7.1.0 as the Amazon EMR release

      iii. Choose Spark interactive as the application bundle

   b. Cluster Configuration

      i. Since we need 1 master node and 4 core nodes, make sure that the primary and core uniform instance groups are configured with m5.xlarge

      ii. remove any task instance groups

   c. Cluster Scaling and Provisioning

      i. Select "Set cluster size manually"

      ii. Next to Core instance groups, change the "Instance Size" value to "4"

   d. Cluster termination and node replacement

      i. Select "Manually terminate cluster" as the termination option

   e. Security Configuration and EC2 key pair

      i. Create a new key pair for the cluster

      ii. Sve the .pem file to access the cluster using SSH

   d. Identity and Access Management (IAM) roles

      i. Set the Service role to EMR_Default Role

      ii. Set the Instance Profile to EMR_EC2_DefaultRole

   e. Click the create cluster button at the bottom of the page


## Accessing the Cluster

1. In AWS Learning Lab, go to the EMR dashboard

2. Select your cluster, and choose "Connect using SSH" under "Cluster Management"

3. Open a new terminal window and use the command "ssh -i ~/pa2.pem hadoop@ec2-3-83-3-228.compute-1.amazonaws.com" to connect

4. Choose "Yes"

5. Set .pem file permissions

  i. chmod 400 /path/to/key/pair.pem

6. Install PySpark & libraries

   i. sudo pip install pyspark

   ii. sudo pip install numpy pandas

7. Write application to the master node



## Run Training Script on EMR Cluster

1. spark-submit

   or

   spark-submit --master yarn --deploy-mode cluster

* NOTE: I ran into significant issues at this step. I was initially trying to access the datasets through the s3 bucket, but couldn't connect to it. I tried changing the IAM roles, but that didn't help either. I then tried to access the files through HFDS but was unsuccessful. I then tried to run the spark job directly on the EMR cluster by setting the HADOOP_CONF_DIR and YARN_CONF_DIR environment variables, but got that error that my Spark environment was missing the LZO compression codec. I tried manually configuring spark to recognize LZO by SSHing into each EC2 instance and installing it but still got errors. I subsequently tried running the script on the master node of the cluster locally without yarn, and added the files locally as well but was ultimately unable to run the code.


# The remainder of these steps are what the overall workflow would have been if I was successful. 


## Run Prediction Script in single EC2 instance without Docker

1. Upload the test script to an EC2 instance

2. spark-submit


## Run Prediction Script with Docker

1. Create a Dockerfile

2. Make sure Docker container has access to Spark

3. Build an image that includes the predction script

4. Deploy image to the same EC2 instance and run it
