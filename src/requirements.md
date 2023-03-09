# Instructions

### [DEADLINE]

Monday - 6th December 2021 (8 weeks)

### [TASK]

1. Implement a feed-forward neural network in C/C++.

2. Train it on a supplied Fashion-MNIST dataset using a backpropagation algorithm.

### [REQUIREMENTS]

Your solution must meet ALL of the following requirements:

1. Your solution must be compilable and runnable on the AISA server.

2. Your solution achieves at least 88% accuracy.

3. Your solution must finish within 30 minutes. (parse inputs, train, evaluate, export results.)

3. Your solution must contain a runnable script called "RUN" (not run, not

RUN.sh, not RUN.exe etc) which compiles, executes and exports the results

into a files.

4. Your solution must output two files:

- "trainPredictions" - network predictions for training input vectors

- "actualTestPredictions" - network predictions for testing input vectors

The format of these files is the same as the supplied training/testing

labels:

- One prediction per line.

- Prediction for i-th input vector (ordered by the input .csv file) must

be on i-th line in the associated output file.

- Each prediction is a single integer 0 - 9.

### [SCORING]

1. All submitted source files will be checked manually.

2. All submitted solutions will also be checked by an automatic evaluator. A similar evaluator is provided to you in the project folder.

3. All submitted solutions will be checked for plagiarism.

4. Any implementation that uses testing input vectors for anything else than evaluation will be awarded 0 points.

5. Presenting someone else's solution (e.g. publicly available solutions on internet) as your own will be awarded 0 points.

3. Use of high-level third-party libraries allowing matrix operations, neural network operations, differentiation operations, linear algebra operations etc is strictly forbidden. Use of such libraries will be awarded 0 points. (Low-level math operations: sqrt, exp, log, rand... is fine)

### [DATASET]

Fashion MNIST (

[https://arxiv.org/pdf/1708.07747.pdf](https://arxiv.org/pdf/1708.07747.pdf)

) a modern version of a

well-known MNIST (

[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

). It is a dataset of Zalando's article images ‒ consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. The dataset is in CSV format. There are four data files included:

- fashion_mnist_train_vectors.csv	- training input vectors

- fashion_mnist_test_vectors.csv	- testing input vectors

- fashion_mnist_train_labels.csv	- training labels

- fashion_mnist_test_labels.csv	- testing labels

### [REMARKS]

1. If you have any questions don't hesitate to contact me!

2. What you do internally with the training dataset is up to you.

3. Write doc-strings.

4. Pack all data with your implementations and put them on the right path so your program will load them correctly on AISA (project dir is fine).

5. You can work alone or you can make teams of two.

6. There are limited number of open spots for solutions written in JAVA. You can enquire about available spots via email (

[422328@mail.muni.cz](mailto:422328@mail.muni.cz)

).

Permissions will be given on a first-come first-served basis. You CANNOT submit solution in JAVA unless you have explicit permission first.

[TIPS]

1. Do not wait until the week before the deadline!

2. Consider using the suggested folder structure (data, src, etc.)

3. Execute your RUN script on AISA before your submission. Missing or non-functional RUN script cannot be evaluated.

4. Do NOT shuffle testing data. It won't fit expected predictions.

5. Solve the XOR problem first. XOR is a very nice example as a benchmark of the working learning process with at least one hidden layer. Btw, the presented network solving XOR in the lecture is minimal and it can be hard to find, so consider more neurons in the hidden layer. If you can't solve the XOR problem, you can't solve Fashion-MNIST.

6. Reuse memory. You are implementing an iterative process, so don't allocate
 new vectors and matrices all the time. An immutable approach is nice but very inappropriate. Also ‒ don't copy data in some cache all the time; use indexes.

7. Objects are fine, but be careful about the depth of object hierarchy you
 are going to create. Always remember that you are trying to be fast.

8. Double precision is fine. You may try to use floats. Do not use BigDecimals or any other high precision objects.

9. Simple SGD is not strong, and fast enough, you are going to need to implement some heuristics as well (or maybe not, but it's highly recommended). I suggest heuristics: momentum, weight decay, dropout. If you are brave enough, you can try RMSProp/AdaGrad/Adam.

10. Start with smaller networks and increase network topology carefully.

11. Consider validation of the model using part of the TRAINING dataset.

12. Adjust hyperparameters to increase your internal validation accuracy.

13. DO NOT WAIT UNTIL THE WEEK BEFORE THE DEADLINE!

### [AISA]

1. .exe files are not runnable on Aisa.

2. Aisa runs on "Red Hat Enterprise Linux Server release 7.5 (Maipo)"

3. Aisa has 4×16 cores, OpenMP or similar easy parallelism may help (in case you use it, please leave some cores to other applications ~ use less than 49 cores)

4. If you are having a problem with missing compilers/maven on Aisa, you can add such tools by including modules

(

[https://www.fi.muni.cz/tech/unix/modules.html.en](https://www.fi.muni.cz/tech/unix/modules.html.en)

). Please, do note, that if your implementation requires such modules, your RUN script must include them as well, otherwise, the RUN script won't work, and I will have no clue what to include.

### [FAQ]

[Q] Can I write in Python, please, please, pretty please?

[A] No. It's too slow without matrix libs.

[Q] Can I implement a convolutional neural network instead of the feed-forward neural network?

[A] Yes, but it might be much harder.

[Q] Can Java implementations compete with C implementations performance-wise?

[A]	Yes. At least one of the best performing implementations was written in java.

Best luck with the project!


# Additional info

### 1. Submission
Prior to submitting your solution into the Homework Vault in the Study Materials
(CZE: Odevzdávarna) please fill in your Student ID (UČO) on the first line in
the README.md file. You can find the README.md inside the project template
directory that you have downloaded from the Study Materials.
This is especially crucial if you're working in groups!

Additional instructions from last year:


We will evaluate your solution as follows:

1. Copy your zip into an empty folder on the Aisa machine
2. unzip *
3. cd *
4. ./RUN
    4a. should add modules if needed
    4b. build your project
    4c. run your project which loads TRAIN/TEST csv files **from the folder as
it was specified in the pv021_project template** => loads
./data/fashion_mnist_test_labels.csv, ./data/fashion_mnist_test_vectors.csv,
./data/fashion_mnist_train_labels.csv, ./data/fashion_mnist_train_vectors.csv
    4d. generate the test predictions as ./actualPredictions file with 10001
lines (should be identical to ./actualPredictionsExample, just with different
numbers per line)

5. Python evaluate
`python3 python_evaluator/evaluate.py ./data/fashion_mnist_test_labels.csv
./actualPredictions`

So, to make sure, it will work for us the same way it works for you, copy your
zip into an empty folder on Aisa and run `unzip * && cd * && ./RUN && python3
python_evaluator/evaluate.py ./data/fashion_mnist_test_labels.csv
./actualPredictions`. If this works, you achieve over 88 % accuracy and the
whole thing takes under 30 minutes, you are can upload your solution into the
Homework Vault.


### 2. Grading and Feedback
In total, you have three attempts to submit your solution - the initial one (by
6th December) and two additional attempts if your solution is returned to you
for correction with our feedback.
For those of you who attended the meeting: this means there are no extra
attempts should you submit your solution before the deadline. So, please, take
your time and verify that your solution is correct and that it meets all the
requirements.

### 3. Requirements
Some of you have misinterpreted the requirements such that your solution should
only work with the FashionMNIST dataset. Note, that the requirements consist of
two tasks:

1) Implementing the feed-forward NN
2) Train it on the FashionMNIST supplied dataset.

Your solution is therefore expected to meet the stated accuracy (at least 88%)
on the FashionMNIST dataset. This condition is necessary but not sufficient.
Your primary objective is to implement a feed-foward neural network. As such
your solution should work with any dataset of a similar complexity (e.g. MNIST
dataset) and the same input/output shape