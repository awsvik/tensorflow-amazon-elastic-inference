
Introduction
------------------------------------
[Amazon Elastic Inference](https://aws.amazon.com/machine-learning/elastic-inference/) allows you to attach low-cost GPU-powered acceleration to [Amazon EC2](http://aws.amazon.com/ec2), [Amazon SageMaker](https://aws.amazon.com/documentation/sagemaker/) and [Amazon ECS](https://aws.amazon.com/ecs/) instances, and reduce the cost of running deep learning inference by up to [75 percent](https://aws.amazon.com/blogs/aws/amazon-elastic-inference-gpu-powered-deep-learning-inference-acceleration/). The `EIPredictor`API makes it easy to use Elastic Inference.

In this workshop, we use the `EIPredictor` and describe a step-by-step example for using TensorFlow with Elastic Inference. We will demostrate how to attach Amzzon Elastic Inference to [Amazon EC2](http://aws.amazon.com/ec2) and Amazon SageMaker](https://aws.amazon.com/documentation/sagemaker/). Additionally, we explore the cost and performance benefits of using Elastic Inference with TensorFlow. We walk you through how we improved total inference time for FasterRCNN-ResNet50 over 40 video frames from ~113.699 seconds to ~8.883 seconds, and how we improved cost efficiency by 78.5 percent.

The `EIPredictor` is based on the [TensorFlow Predictor](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/predictor) API. The `EIPredictor` is designed to be consistent with the TensorFlow Predictor API to make code portable between the two data structures. The `EIPredictor` is meant to be an easy way to use Elastic Inference within a single Python script or notebook. A flow that’s already using the TensorFlow Predictor only needs one code change: importing and specifying the`EIPredictor`. This procedure is shown later.

Benefits of Amazon Elastic Inference
------------------------------------

Look at how Elastic Inference compares to other EC2 options in terms of performance and cost.

![](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2019/07/08/EIPredictor-1.gif)

|Instance Type|vCPUs|CPU Memory (GB)|GPU Memory (GB)|FP32 TFLOPS|$/hour|TFLOPS/$/hr|
|---|---|---|---|---|---|---|
|1|m5.large|2|8|–|0.07|$0.10|0.73|
|2|m5.xlarge|4|16|–|0.14|$0.19|0.73|
|3|m5.2xlarge|8|32|–|0.28|$0.38|0.73|
|4|m5.4xlarge|16|64|–|0.56|$0.77|0.73|
|5|c5.4xlarge|16|32|–|0.67|$0.68|0.99|
|6|p2.xlarge (K80)|4|61|12|4.30|$0.90|4.78|
|7|p3.2xlarge (V100)|8|61|16|15.70|$3.06|5.13|
|8|eia.medium|–|–|1|1.00|$0.13|7.69|
|9|eia.large|–|–|2|2.00|$0.26|7.69|
|10|eia.xlarge|–|–|4|4.00|$0.52|7.69|
|11|m5.xlarge + eia.xlarge|4|16|4|4.14|$0.71|5.83|

If you look at compute capability (teraFLOPS or floating point operations per second), m5.4xlarge provides 0.56 TFLOPS for $0.77/hour, whereas an eia.medium with 1.00 TFLOPS costs just $0.13/hour. If pure performance (ignoring costs) is the goal, it’s clear that a p3.2xlarge instance provides the most compute at 15.7 TFLOPS.

However, in the last column for TFLOPS per dollar, you can see that Elastic Inference provides the most value. Elastic Inference accelerators (EIA) must be attached to an EC2 instance. The last row shows one possible combination. The m5.xlarge + eia.xlarge has a similar amount of vCPUs and TFLOPS as a p2.xlarge, but at a $0.19/hour discount. With Elastic Inference, you can right-size your compute needs by choosing your compute instance, memory and GPU compute. With this approach, you can realize the maximum value per $ spent. The GPU attachments to your CPU are abstracted by framework libraries, which makes it easy to make inference calls without worrying about the underlying GPU hardware.


Hands-on Workshop
------------------------------------

## Lab 1: Attach Amazon Elastic Inference to Amazon SageMaker Inference Endpoint


### Step 1: Create an Amazon SageMaker Notebook Instance

An Amazon SageMaker notebook instance is a fully managed machine learning (ML) Amazon Elastic Compute Cloud (Amazon EC2) compute instance that runs the Jupyter Notebook App. You use the notebook instance to create and manage Jupyter notebooks that you can use to prepare and process data and to train and deploy machine learning models. 

Note:
If necessary, you can change the notebook instance settings, including the ML compute instance type, later.

**To create an Amazon SageMaker notebook instance**

1.  Open the Amazon SageMaker console at [https://console.aws.amazon.com/sagemaker/](https://console.aws.amazon.com/sagemaker/).
    
2.  Choose **Notebook instances**, then choose **Create notebook instance**.
    
3.  On the **Create notebook instance** page, provide the following information (if a field is not mentioned, leave the default values):
    
    1.  For **Notebook instance name**, type a name for your notebook instance.
        
    2.  For **Instance type**, choose `ml.t2.medium`. This is the least expensive instance type that notebook instances support, and it suffices for this exercise.
        
    3.  For **IAM role**, choose **Create a new role**, then choose **Create role**.
        
    4.  Choose **Create notebook instance**.
        
        In a few minutes, Amazon SageMaker launches an ML compute instance—in this case, a notebook instance—and attaches an ML storage volume to it. The notebook instance has a preconfigured Jupyter notebook server and a set of Anaconda libraries.
        

Next Step
___

### Step 2: Open Terminal Window on Amazon SageMaker Jupyter Notebook Instance


1.  Open the notebook instance.
    
    1.  Sign in to the Amazon SageMaker console at [https://console.aws.amazon.com/sagemaker/](https://console.aws.amazon.com/sagemaker/).
        
    2.  Open the notebook instance, by choosing either **Open Jupyter** for classic Juypter view or **Open JupyterLab** for JupyterLab view next to the name of the notebook instance. The Jupyter notebook server page appears:
        
2.  Open Terminal Window.
    
    1.  If you opened the notebook in Jupyter classic view, on the **Files** tab, choose **New**, and **Terminal** at the bottom of the menu. 
    2. In ther terminal console execute ``sh-4.2$cd SageMaker/``

    
Next Step
___
### Step 3: Download Ipython Notebook from Git Repo.


1.  In the terminal console execute In ther terminal console execute ``sh-4.2$ git clone <git repo>``
2. Swtich to the **Home** tab in your browser window. 
3. In the SageMaker Jupyter Notebook Instance console, in the Files tab, navigate to ``tensorflow-amazon-elastic-inference\lab1\TFWorld-Amazon-Elastic-Inference-Lab1.ipynb``
4. Click on ``TFWorld-Amazon-Elastic-Inference-Lab1.ipynb`` and follow the instruction in Jupyter Notebook to complete Lab 1. After completing Lab 1, come back, we will use this same Jupyter Notebook Instance to get started on Lab 2. In Lab 1, you will train a model and deploy it on Amazon SageMaker with Amazon Elastic Inference Attached and without Amazon Elastic Inference Attached. Note the latency and cost difference in with and wihtout Amazon Elastic Inference option. Don't focus too much on undertanding training. Breeze through the training part quickly. 

## Lab 2: Attach Amazon Elastic Inference to Amazon EC2


### Video object detection example using the EIPredictor on Amazon Ec2
---

Here is a step-by-step example of using Elastic Inference with the `EIPredictor`. For this example, we use a FasterRCNN-ResNet50 model, an m5.large CPU instance, and an eia.large accelerator.

### Create Key Pair for Remote Ec2 Instance with DL AMI
1. Switch to the Amazon SageMaker terminal console in your browser tab where you ran ``sh-4.2$ git clone https://github.com/awsvik/tensorflow-amazon-elastic-inference.git``
2. In ther terminal console execute ``sh-4.2$ source activate python3``. Python 3 is required to run the script that is avaliable in lab 2 folder of git repo.
3. Create your key pair using the Amazon EC2 console

4.  Open the Amazon EC2 console at [https://console.aws.amazon.com/ec2/](https://console.aws.amazon.com/ec2/).
    
5.  In the navigation pane, under **NETWORK & SECURITY**, choose **Key Pairs**.
    
    Note:
    The navigation pane is on the left side of the Amazon EC2 console. If you do not see the pane, it might be minimized; choose the arrow to expand the pane.
    
6.  Choose **Create Key Pair**.
    
7.  For **Key pair name**, enter a name for the new key pair, and then choose **Create**.
    
8.  The private key file is automatically downloaded by your browser. The base file name is the name you specified as the name of your key pair, and the file name extension is `.pem`. Save the private key file in a safe place. This is the only chance for you to save the private key file. You'll need to provide the name of your key pair when you launch an instance and the corresponding private key each time you connect to the instance.
    
9.  If you will use an SSH client on a Mac or Linux computer to connect to your Linux instance, use the following command to set the permissions of your private key file so that only you can read it.
    
    ``chmod 400 `my-key-pair`.pem``
    
    If you do not set these permissions, then you cannot connect to 	your instance using this key pair. For more information, see 	[Error: Unprotected Private Key File]	(TroubleshootingInstancesConnecting.html#troubleshoot-unprotected-	key).
    
	### Use EI setup tool to launch Amazon Ec2 instance with DL AMI and Amazon EI attached

10. We use the Amazon Elastic Inference (EI) setup tool which is a Python script that enables you to quickly get started with EI. 

	**change working directory to lab2**
``$cd /home/ec2-user/SageMaker/tensorflow-amazon-elastic-inference/lab2``
**run the script**
``$python amazonei_setup.py --region <AWS-Region-Name> --instance-type m5.xlarge``

	The script will ask you to choose following

    a. Choose Operating System 
    b. Choose Accelerator size
    c. Choose VPC
    d. Launch an instance.   
    
  At a high level, the script does the following:
	
	a. Creates an IAM role for the instance with an IAM policy that lets you connect to the AWS Elastic Inference service.
    b. Creates a security group with the necessary ingress and egress rules to allow the instance to communicate with the accelerator.
    c. Creates an AWS PrivateLink VPC Endpoint within your desired subnet.
    d. Launches the desired EC2 instance with an EI accelerator using the latest AWS Deep Learning AMI (DLAMI) for the chosen operating system

	Launch and wait for the instance to reach running state.

	Waiting for instance to reach running state ...

	**Note**: Please wait until instance is fully initialized and ready to accept SSH connections. You may check instance status at EC2 console.Also please locate your private key file <key-pair>.pem

	``Type 'y' to continue. Type 'q' to quit.``
``amazon-elastic-inference-tools $``

	Type 'q' to quit if your ec2 instance is running in Amazon EC2 console.

11. Locate your keypair.pem on your personal laptop. Navigate to the SageMaker Jupyter Notebook Instance console in your browser, in Jupyter Notebook Files tab, navigate to ``SageMaker\tensorflow-amazon-elastic-inference\lab2``. Look for ``upload`` button on the Jupyter Notebook(LHS). Use this button to upload your keypair.pem file from your personal laptop.

12. In your browser tab, navigate to SageMaker Jupyter Notebook Terminal window and SSH into the EC2 instance that was created by Amazon Elastic Inference (EI) setup tool.
Changer permissions on my-key-pair.pem

	``chmod 400 `my-key-pair`.pem``

	For Ubuntu:
	``ssh -i {keypair.pem} ubuntu@{ec2 instance public DNS name}``

	 For Amazon Linux:
	``ssh -i {keypair.pem} ec2-user@{ec2 instance public DNS name}``

13.  In the remote EC2 instance(DL AMI with EI attached), copy the code.
``git clone https://github.com/aws-samples/aws-elastic-inference-tensorflow-examples``
    
14.  Build and run your Jupyter notebook on remote instance.
    
 ``cd aws-elastic-inference-tensorflow-examples; 
 ./build_run_ei_container.sh``
    
    Wait until the Jupyter notebook starts up. 
    
15. Inorder to access the Jupyter Notebook running on remote EC2 	instance, SSH into remote Ec2 instance from the terminal or SSH 	client on your personal laptop with port forwarding for the Jupyter 	notebook.

 	For Ubuntu AMIs:
    
        ``ssh -i {/path/to/keypair} -L 8888:localhost:8888 ubuntu@{ec2 	instance public DNS name}``
    
   For Amazon Linux AMIs:
    
        ``ssh -i {/path/to/keypair} -L 8888:localhost:8888 ec2-	user@{ec2 instance public DNS name}``
 
16. Go to localhost:8888 and supply the token that is given in the terminal. In the terminal you will see a ``http://localhost:8888/<token>`` url with token. 

	### Access your Jupyter Notebook from your personal laptop broswer window

17.  Run benchmarked versions of Object Detection examples.
    1.  Open `elastic_inference_video_object_detection_tutorial.ipynb` and run the notebook.
    2.  Take note of the session runtimes produced. The following two examples show without Elastic Inference, then with Elastic Inference.
        1.  The first is TensorFlow running your model on your instance’s CPU, without Elastic Inference:
            
                Model load time (seconds): 8.36566710472
                Number of video frames: 40
                Average inference time (seconds): 2.86271090508
                Total inference time (seconds): 114.508436203
            
        2.  The second reporting is using an Elastic Inference accelerator:
            
                Model load time (seconds): 21.4445838928
                Number of video frames: 40
                Average inference time (seconds): 0.23773444891
                Total inference time (seconds): 9.50937795639
            
    3.  Compare the results, performance, and cost between the two runs.
        *   In the screenshots posted above, Elastic Inference gives an average inference speedup of ~12x.
        *   With this video of 340 frames of shape (1, 1080, 1920, 3) simulating streaming frames, about 44 of these full videos can be inferred in one hour using the m5.large+eia.large, considering one loading of the model.
        *   With the same environment excluding the eia.large Elastic Inference accelerator, only three or four of these videos can be inferred in one hour. Thus, it would take 12–15 hours to complete the same task.
        *   An m5.large costs $0.096/hour, and an eia.large slot type costs $0.26/hour. Comparing costs for inferring 44 replicas of this video, you would spend $0.356 to run inference on 44 videos in an hour using the Elastic Inference set up in this example. You’d spend between $1.152 and $1.44 to run the same inference job in 12–15 hours without the eia.large accelerator.
        *   Using the numbers above, if you use an eia.large accelerator, you would run the same task in between a 1/12th and a 1/15th of the time and at ~27.5% of the cost. The eia.large accelerator allows for about 4.2 frames per second.
        *   The complete video is 340 frames. To run object detection on the complete video, remove  `and count < 40` from the `def extract_video_frames` function.
    4.  Finally, you should produce a video like this one: [annotated\_dog\_park.mp4](https://aws-ml-blog.s3.amazonaws.com/artifacts/optimizing-costs-ei/annotated_dog_park.mp4).
    5.  Also note the usage of the `EIPredictor` for using an accelerator (`use_ei=True`) and running the same task locally (`use_ei=False`).
        
            ei_predictor = EIPredictor(
                            model_dir=PATH_TO_FROZEN_GRAPH,
                            input_names={"inputs":"image_tensor:0"},
                            output_names={"detections_scores":"detection_scores:0",
                                          "detection_classes":"detection_classes:0",
                                          "detection_boxes":"detection_boxes:0",
                                          "num_detections":"num_detections:0"},
                            use_ei=True)
            
        
### Exploring all possibilities

Now, we’ve done more investigation and tried out a few more instance combinations for Elastic Inference. We experimented with FasterRCNN-ResNet50, batch size of 1, and input image dimensions of (1080, 1920, 3).

The model is loaded into memory with an initial inference using a random input of shape (1, 100, 100, 3). After rerunning the initial notebook, we started with combinations of m5.large, m5.xlarge, m5.2xlarge, and m5.4xlarge with Elastic Inference accelerators eia.medium, eia.large, and eia.xlarge. We produced the following table:

-|A|B|C|D|E|
|---|---|---|---|---|---|
|1|Client instance type|Elastic Inference accelerator type|Cost per hour|Infer latency [ms]|Cost per 100k inferences|
|2|m5.large|eia.medium|$0.23|<span style="color:red"></style>353.53|$2.22|
|3||eia.large|$0.36|222.78|$2.20|
|4||eia.xlarge|$0.62|<span style="color:blue"></style>140.96|$2.41|
|5|m5.xlarge|eia.medium|$0.32|<span style="color:red"></style>357.70|$3.20|
|6||eia.large|$0.45|224.81|$2.82|
|7||eia.xlarge|$0.71|<span style="color:blue"></style>150.29|$2.97|
|8|m5.2xlarge|eia.medium|$0.51|<span style="color:red"></style>350.38|$5.00|
|9||eia.large|$0.64|229.65|$4.11|
|10||eia.xlarge|$0.90|<span style="color:blue"></style>142.55|$3.58|
|11|m5.4xlarge|eia.medium|$0.90|<span style="color:red"></style>355.53|$8.87|
|12||eia.large|$1.03|222.53|6.35|
|13||eia.xlarge|$1.29|<span style="color:blue"></style>149.17|$5.34|



Looking at the client instance types with the eia.medium (highlighted in red in the table above), you see similar results. This means that there isn’t much client-side processing, so going to a larger client instance does not improve performance. You can save on cost by choosing a smaller instance.

Similarly, looking at client instances using the largest eia.xlarge accelerator (highlighted in blue), there isn’t a noticeable performance difference. This means that you can stick with the m5.large client instance type, achieve similar performance, and pay less. For information about setting up different client instance types, see [Launch accelerators in minutes with the Amazon Elastic Inference setup tool for Amazon EC2](https://aws.amazon.com/blogs/machine-learning/launch-ei-accelerators-in-minutes-with-the-amazon-elastic-inference-setup-tool-for-ec2/).

#### Comparing M5, P2, P3, and EIA instances

Plotting the data that you’ve collected from runs on different instance types, you can see that GPU performed better than CPU (as expected). EC2 P3 instances are 3.34x faster than EC2 P2 instances. Before this, you had to choose between P2 and P3. Now, Elastic Inference gives you another choice, with more granularity at a lower cost.

Based on instance cost per hour (us-west-2 for [EIA](https://aws.amazon.com/machine-learning/elastic-inference/pricing/) and [EC2](https://aws.amazon.com/ec2/pricing/on-demand/)), the m5.2xlarge + eia.medium costs in between the P2 and P3 instance costs (see the following table) for the TensorFlow `EIPredictor` example. When factoring the cost to perform 100,000 inferences, you can see that the P2 and P3 have a similar cost, while with m5.large+eia.large, you achieve nearly P2 performance at less than half the price!

|-|A|B|C|D|
|---|---|---|---|---|
|1|Instance Type|Cost per hour|Infer latency [ms]|Cost per 100k inferences|
|2|m5.4xlarge|$0.77|415.87|$8.87|
|3|c5.4xlarge|$0.68|363.45|$6.87|
|4|p2.xlarge|$0.90|197.68|$4.94|
|5|p3.2xlarge|$3.06|61.04|$5.19|
|6|m5.large+eia.large|$0.36|222.78|$2.20|
|7|m5.large+eia.xlarge|$0.62|140.96|$2.41|


![](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2019/07/08/EIPredictor-2.gif)

### Comparing inference latency

Now that you’ve decided on an m5.large client instance type, you can look at the accelerator types (the orange bars). There is a progression from 222.78 ms and 140.96 ms in terms of inference latency. This shows that the Elastic Inference accelerators provide options between P2 and P3 in terms of latency, at a lower cost.

### Comparing inference cost efficiency

The last column in the preceding table, Cost per 100k inferences, shows the cost efficiency of the combination. m5.large and eia.large have the best cost efficiency. The m5.large + eia.large combo provides the best cost efficiency compared to the m5.4xlarge and P2/P3 instances with 55% to 75% savings.

The m5.large and eia.xlarge provides a 2.95x speed increase over m5.4xlarge (CPU only) with 73% savings and a 1.4x speedup over p2.xlarge with 51% savings.

Results
-------

Here’s what we’ve found so far:

*   Combining Elastic Inference accelerators with any client EC2 instance type enables users to choose the amount of client compute, memory, etc. with a configurable amount of GPU memory and compute.
*   Elastic Inference accelerators provide a range of memory and GPU acceleration options at a lower cost.
*   Elastic Inference accelerators can achieve a better cost efficiency than M5, C5, and P2/P3 instances.

In our analysis, we found that increasing ease of use within TensorFlow is as simple as creating and calling an `EIPredictor` object. This allowed you to use largely the same test notebook on CPU, GPU, and CPU+EIA environments with TensorFlow, and ease testing and performance analysis.

We started with a FasterRCNN-ResNet50 model running on an m5.4xlarge instance with a 415.87 ms inference latency. We were able to reduce it to 140.96 ms by migrating to an m5.large and eia.xlarge, resulting in a 2.95x increase in speed with a $0.15 hourly cost savings to top it off. We also found that we could achieve a $0.41 hourly cost savings with an m5.large and eia.large and still get better performance (416 ms vs. 223 ms).

Conclusion
----------

Try out TensorFlow on Elastic Inference and see how much you can save while still improving performance for inference on your model. Here are the steps we went through to analyze the design space for deep learning inference, and you too can follow for your model:

1.  Write a test script or notebook to analyze inference performance for CPU context.
2.  Create copies of the script with tweaks for GPU and EIA.
3.  Run scripts on M5, P2, and P3 instance types and get a baseline for performance.
4.  Analyze the performance.
    1.  Start with the largest Elastic Inference accelerator type and large client instance type.
    2.  Work backwards until you find a combo that is too small.
5.  Introduce cost efficiency to the analysis by computing cost to perform 100k inferences.

* * *