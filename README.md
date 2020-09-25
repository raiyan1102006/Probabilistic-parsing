# Probabilistic Parsing using Viterbi CYK Decoder

**Author:** Raiyan Abdul Baten

This project was done as part of a homework in the CSC448 Statistical Speech and Language Processing course at the University of Rochester.

## Task
**Part a:** implement a viterbi cyk decoder using the following features:

rule: (A,B,C)

first word: (A, xi+1 )

last word: (A, xk )

A test case and description of the file format is given in the a.txt file.

**Part b:** implement perceptron training for this set of features. See test case in the b.txt file.

## Usage Instruction

To run the programs, simply use the following commands:

(a)
```
python3 parser.py weights test
```
(b)
```
python3 perceptron.py train weights.trained
```


## Acknowledgment
I took help from this set of slides, where the weighted CYK algorithm is explained nicely: 
http://www.cs.virginia.edu/~kc2wc/teaching/NLP16/slides/16-CKY.pdf
