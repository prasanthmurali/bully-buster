1. Download good_corpus.txt.gz and bad_corpus.txt.gz from 
https://github.com/HackHarassment/TwitterClassifier/blob/master/data/good_corpus.txt.gz and
https://github.com/HackHarassment/TwitterClassifier/blob/master/data/bad_corpus.txt.gz respectively.
Unzip the files into "good_corpus.txt" and "bad_corpus.txt" and place them inside the Submissions folder.
(The Submissions.zip is submitted, unzipping it will give Submissions folder.)
These files contain the non abusive and abusive corpora respectively. 
The other files needed for the project to run are already present in the 
Submissions folder. 

2. The file "Test_Data.txt" contains the testing corpora. 

2. Open Project.py and run the code. 

3. The three corpora are analyzed and the unigram features are written on to "good_counts.txt",
"bad_counts.txt" and "test_counts.txt".

4. The probability counts of unigram features of bad and good corpora are written onto "good_counts_prob.txt"
and "bad_counts_prob.txt".

5. The Naive Bayes results are written on to "Naive_Bayes_Model_Results.txt".

6. The actual classifications of test tweets are written in "Test_Answers.txt", with each line number
corresponding to the tweet number.

7. The Precision Recall analysis is performed on the Naive_Bayes model and the curve is saved as PR Curve.jpg. 
8. Post this, LDA starts. 

9. LDA is performed on good and bad corpora and saved on to "good_LDA.txt" and "bad_LDA.txt".

10. LDA is also performed on "Test_LDA_bad.txt" - that contains abusive tweets and "Test_LDA_good.txt" - that
contains non abusive tweets and the results saved on to "Test_bad_LDA.txt" and "Test_good_LDA.txt" respectively.

11. The results of LDA model on these test files are saved on to "LDA_results.txt".

NOTE: Since running the program might take a lot of time, the solution files are all submitted as 
      a part of the project itself.
      Running the program will create the same set of output files with same values in them. 

The initial parts of the project can also be found on devpost at https://devpost.com/software/bully-buster