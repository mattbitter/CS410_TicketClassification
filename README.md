## Synopsis
This script predicts the correct IT service desk ticket classification based on text data from a Canadian CPG company.
A number of models were tested with TF-IDF unigrams SVM being the best.

## Install/Setup
[Anaconda](https://www.anaconda.com/download/) is a great way to get started with Python, and is recommended to utilize this program.  It contains all the libraries necessary to run the script as well as the Jupyter Notebook feature to execute the code in sections for an optimal experience.
Using git, pull the package to a directory of your choice.  The program package is located [here](https://github.com/mattbitter/CS410_TicketClassification/)

## Start the program
1. Run Jupyter Notebook
	*  This can be located in the Windows Start menu within the Anaconda folder.
	*  This should launch you into your default web browser and open the Jupyter Notebook server interface.
2. Navigate to location where project is downloaded on your computer.
3. Within the directory where you pulled the project, launch ‘jn_star_main’ and it will bring up the Jupyter Notebook and the program in its logical blocks.
	* N.B. the Repo also contains a regular python file which is a duplicate of the notebook. It is recommended to use the Notebook if it is your first time running the script.

## Running the script
4. Within Jupyter, there are logical blocks of code that are grouped for the user's convenience.  The sections should be run in the following order (counting sections from the top):

Sec 4.1 – Import Libraries and setup the program for execution.
	* KEY NOTE:  Uncomment the corpus download lines on first run so that all of the data is retrieved for the program.  After the initial run it is advised to comment these out again to ensure the fastest run time.
Sec 4.2 – Import the collection of documents (our ticket data) and clean that data for use in our modelling
	* tokenizing, lemming, stemming, Removal of punctuation, Removal of numbers
	* Removal of Support Groups not pertinent to the classification
	* Lastly this section creates the training and testing data sets (20% testing set default)
Sec 4.3 – Training and fitting the model
	*  SVM model utilized
	* Implement TF-IDF weighting with sublinear_tf set true 
	* To improve performance the Support Group headers are converted to unique numeric values 
	* Testing and prediction performed here
Sec 4.4 – Evaluation – Overall the final F1 score for this model was 85%.  This section includes the following eveluation objects:
	*  Confusion Matrix (**visualization**), Precision, Recall, Micro, Macro and Weighted F1 Scoring
Sec 4.5 – User Testing (your turn to play with it!)
	* Users can update the ‘str_new’ variable in the section to see how their inputs are classified in the ticketing support system

## Optimizing parameters
	5. It is important to seelct the optimal C value when using SVM. Accordingly, grid search and stratified k fold cross validation were used to ensure the parameters were correct. Section 5.1 does not need to be run by the user
		Sec 5.1 - cross validation identified that the true accuracy of our model is around 85% with ~2.5% std deviation.
		Sec 5.2 - Grid search. It found that C = 3 was the best.

## Other models
	6. The models shown after Section 5 did not perform as accuractly compared to the simple SVM TF-IDF model:
		Sec 6.1 – Decision Trees – only produced a 70% F1 and Naïve Bayes produced ~50% F1.  Not a top performer for the ticket classification system.
		Sec 6.2 – LDA and NMF approach (You will need to update section 4.3 under the ‘#fit’ comment section to change the ‘probability’ variable to ‘True’.)
			i. utilized LDA and NMF features to form a dense matrix of topic probabilities based off of the training TF-IDF values. 15 Topics were chosen as it showed the best results. This LDA and NMF dense matrix was merged with probabilities of the classes predicted from the SVM model and fed into a second SVM model.
			ii. NMF appeared to perform better v.s. LDA because NMF is able to leverage TF-IDF while LDA only uses TF. 
			iii. In all the end resulted in a 1% increase in cross validated F1 so it is arguably not worth the complexity.
		Sec 6.3 – Used individual users names as one-hot features to add into the SVM model using a similar approach to Section 6.2. However, it shows no changes in accuracy score.
			i.	You will need to update section 3 under the ‘#fit’ comment section to change the ‘probability’ variable to ‘True’.
