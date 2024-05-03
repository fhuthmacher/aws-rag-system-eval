# RAG System Component Evaluation
    
This notebook is a follow up from the previous [notebook](https://github.com/fhuthmacher/LLMevaluation/blob/main/LLMInformationExtraction.ipynb) and [youtube video](https://www.youtube.com/watch?v=HUuO9eJbOTk) in which we explored the overall evaluation approach and a RAG system's overall accuracy.

This notebook we will take a closer look at specific RAG evaluation metrics and explore how different RAG components impact these RAG evaluation metrics.


![Solution Architecture](/images/architecture.png)

## Instructions
### 1. Launch CloudFormation Stack
If you want to deploy the entire solution with IaC, log into your AWS management console. Then click the Launch Stack button below to deploy the required resources.

[![Launch CloudFormation Stack](https://felixh-github.s3.amazonaws.com/misc_public/launchstack.png)](https://console.aws.amazon.com/cloudformation/home#/stacks/new?stackName=rageval&templateURL=https://felixh-github.s3.amazonaws.com/misc_public/rageval.yml)

To avoid incurring additional charges to your account, stop and delete all of the resources by deleting the CloudFormation template at the end of this tutorial.

### 2. Amazon Bedrock Foundation Model Access Configuration
Amazon Bedrock offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon. In the Amazon Bedrock console, you can see the descriptions for these foundation models by the Base models link in the navigation pane.

You need to request access to these foundation models before using them. In the Amazon Bedrock console, do the following:

Go to the Amazon Bedrock console. 
In the navigation pane, select Model access.
In the Model access page, you should see a list of base models. Click on the Manage model access button to edit your model access.
Select the check box next to all of the following models to get started.
- Titan
- Claude
- Cohere

![Amazon Bedrock Model Access](/images/modelaccess.png)

Click on the Save changes button and it may take several minutes to save the changes. This also brings you back to the Model access page.
Models will show as Access granted on the Model access page under the Access status column, if access is granted.

### 3. Log into SageMaker Console
To access VSCode Editor hosted in SageMaker, go to [Amazon SageMaker console](https://us-east-1.console.aws.amazon.com/sagemaker/home), select Domains in the navigation menu, and click on the newly created domain "SageMakerDomain". 
![SageMaker Domain](/images/SageMakerDomain.png)

### 4. Launch SageMaker Studio
Launch SageMaker Studio by selecting "Studio" from the dropdown button on the right.
![SageMaker Studio Launch](/images/SageMakerStudioLaunch.png)

### 5. Open SageMaker CodeEditor
Once in SageMaker Studio, first select CodeEditor in the top left, and then click on the "open" link as shown below.
![Amazon SageMaker Code Editor Access](/images/CodeEditorAccess.png)

### 6. Clone Git Repository
In CodeEditor, open a terminal and run the following command:
git clone https://github.com/fhuthmacher/aws-rag-system-eval

### 7. Review dev-rageval.env
Review dev-rageval.env file and make updates as needed.

### 8. Run Jupyter notebook 
In CodeEditor, open Jupyter notebook RAG-System-Eval-DeepDive-mlflow.ipynb and run all cells