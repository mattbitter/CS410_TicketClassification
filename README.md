# CS410_TicketClassification
Install/Setup
1. Anaconda is a great way to get started with Python, and is recommended to utilize this program.  It contains all the libraries necessary to run the script as well as the Jupyter Notebook feature to execute the code in sections for an optimal experience.
2. Using git, pull the package to a directory of your choice.  The program package is located at:  https://github.com/mattbitter/CS410_TicketClassification/

Start the program
1. Run Jupyter Notebook
	a.  This can be located in the Windows Start menu within the Anaconda folder.
	b.  This should launch you into your default web browser and open the Jupyter Notebook server interface.
2. Navigate to location where project is downloaded on your computer.
3. Within the directory where you pulled the project, launch ‘jn_star_main’ and it will bring up the Jupyter Notebook and the program in its logical blocks.
N.B. the Repo also contains a regular python file which is a duplicate of the notebook. It is recommended to use the Notebook if it is your first time running the script.

Running the script
1. Within Jupyter, there are logical blocks of code that are grouped for the user's convenience.  The sections should be run in the following order (counting sections from the top):
  a.	Sec.1 – Import Libraries and setup the program for execution.
		i.	KEY NOTE:  Uncomment the corpus download lines on first run so that all of the data is retrieved for the program.  After the initial run it is advised to comment these out again to ensure the fastest run time.
	b.	Sec.2 – Import the collection of documents (our ticket data) and clean that data for use in our modelling
		i.  tokenizing
		ii. lemming
		iii.stemming
		iv. Removal of punctuation
		v.  Removal of numbers
		vi. Removal of Support Groups not pertinent to the classification
		vii.Lastly this section creates the training and testing data sets (20% testing set default)
	c.	Sec.3 – Training and fitting the model
		i.  SVN model utilized
		ii. Implement TF-IDF weighting with sublinear_tf set true 
		iii.To improve performance the Support Group headers are converted to unique numeric values 
		iv. Testing and prediction performed here
	d.	Sec.4 – Evaluation – Overall the final F1 score for this model was 85%.  This section includes the following eveluation objects:
		i.  Confusion Matrix
		ii. Precision
		iii.Recall
		iv. F1 Scoring
	e.	Sec.5 – User Testing (your turn to play with it!)
		i.	Users can update the ‘str_new’ variable in the section to see how their inputs are classified in the ticketing support system

Other fun things to try and cool experiements performed (and you can try them too!):
1.	These sections (6-10) include various tests and models that were tested in building this model.  Follow the directions in each section to repeat the tests at your own discretion. 
2.	Section 6 – Cross validation testing - you should be able to run this section as is to see the results it produces.
3.	Section 7 – Grid search cross validation – provides the optimal parameters for the models tested.  Plug in a model and it will help you optimize it.
4.	Section 8 – Decision Trees – only produced a 70% F1 and Naïve Bayes produced ~50% F1.  Not a top performer for the ticket classification system.
5.	Section 9 – LDA approach – utilized LDA and NMF features to form dense matrix and fed into a new SVM with the new features.  Not optimal but interesting!  NMF appeared to perform better with a lot more effort – but in the end only increased F1 by ~1%.
	a.	You will need to update section 3 under the ‘#fit’ comment section to change the ‘probability’ variable to ‘True’.
6.	Section 10 – Experimental section using SVN model to attempt to get F1 above 90% - which it did not.  Tried adding in user names and team names – but no improvement in the model.
	a.	You will need to update section 3 under the ‘#fit’ comment section to change the ‘probability’ variable to ‘True’.
