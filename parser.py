#!/usr/bin/python3

import numpy as np
import itertools
import sys

def read_weights_file(weights_list):
	grammar = {'T':{},
			   'R':{},
			   'F':{},
			   'L':{}}

	symbols = {'pre_terminals' : set(),
			  'non_terminals' : set(),
			  'terminals' : set()}

	for a_line in weights_list:
		a_line = a_line.strip().split(' ')
		rule = a_line[0]
		weight = a_line[1]
		rule_list = rule.split("_")

		if rule_list[0]=='T': #preterminals
			grammar[rule_list[0]][(rule_list[1],rule_list[2])] = int(weight)
			symbols['pre_terminals'].add(rule_list[1])
			symbols['terminals'].add(rule_list[2])

		elif rule_list[0] in ['F','L']: #first and last word features
			grammar[rule_list[0]][(rule_list[1],rule_list[2])] = int(weight)
			symbols['non_terminals'].add(rule_list[1])
			symbols['terminals'].add(rule_list[2])

		elif rule_list[0]=='R': #rule features
			grammar[rule_list[0]][(rule_list[1],rule_list[2],rule_list[3])] = int(weight)
			symbols['non_terminals'].add(rule_list[1])
   
	return grammar, symbols


def build_tree(bckptr,delta,word_list,init):
	left_node,right_node,split_point = bckptr[init[0]][init[1]][init[2]]
	tree_string_snippet = "("+str(init[0])+" " #recursively build this up
	
	# left branch
	if split_point-init[1]==1: #reached leaf node
		node_str = "("+str(left_node)+" "+str(word_list[init[1]])+")"
		tree_string_snippet += node_str
	else:
		tree_string_snippet += build_tree(bckptr,delta,word_list,(left_node,init[1],split_point))
	
	# right branch
	if init[2]-split_point==1: #reached leaf node
		node_str = "("+str(right_node)+" "+str(word_list[init[2]-1])+")"
		tree_string_snippet += node_str
	else:
		tree_string_snippet += build_tree(bckptr,delta,word_list,(right_node,split_point,init[2]))
	
	tree_string_snippet += ")" 
	return tree_string_snippet

def viterbi_cyk(word_list,grammar,symbols):
	N = len(word_list)
	tags = list(symbols['pre_terminals'].union(symbols['non_terminals']))
	
	#the first columns in delta and bckptr are useless, exist for convenient indexing
	delta = {key:np.ones((N,N+1), dtype = 'object')*-1*np.inf for key in tags} 
	bckptr = {key:np.ones((N,N+1), dtype = 'object')*-1*np.inf for key in tags} 
	
	#compute missing pre-terminal weights (T)
	for pre_terminal in symbols['pre_terminals']:
		for terminal in symbols['terminals']:
			if (pre_terminal,terminal) not in grammar['T']:
				grammar['T'][(pre_terminal,terminal)] = 0
	
	#compute missing rule weights (R)  
	x = symbols['pre_terminals'].union(symbols['non_terminals'])
	x.discard('S') #RHS won't have the S symbol

	for LHS in symbols['non_terminals']: #LHS doesn't have pre-terminals
		for RHS in [p for p in itertools.product(x, repeat=2)]:
			if (LHS,RHS[0],RHS[1]) not in grammar['R']:
				grammar['R'][(LHS,RHS[0],RHS[1])] = 0
		
	#compute missing first and last-word weights (F, L)
	for LHS in symbols['non_terminals']: #LHS doesn't have pre-terminals
		for RHS in symbols['terminals']:
			if (LHS,RHS) not in grammar['F']:
				grammar['F'][(LHS,RHS)] = 0
			if (LHS,RHS) not in grammar['L']:
				grammar['L'][(LHS,RHS)] = 0
	
	# viterbi cyk
	for i in range(1,N+1):
		for pre_terminal in symbols['pre_terminals']:
			delta[pre_terminal][i-1][i] = grammar['T'][(pre_terminal,word_list[i-1])]
	   
	for width in range(2,N+1):
		for start in range(N-width+1):
			end = start+width
			for mid in range(start+1,end):
				for rule,weight in grammar['R'].items():
					prev_weight = delta[rule[0]][start][end]
					current_weight = delta[rule[1]][start][mid]\
									+delta[rule[2]][mid][end]+grammar['R'][rule]+\
									grammar['F'][(rule[0],word_list[start])]+\
									grammar['L'][(rule[0],word_list[end-1])]
					delta[rule[0]][start][end] = max(prev_weight,current_weight)
					if current_weight>prev_weight:
						bckptr[rule[0]][start][end] = (rule[1],rule[2],mid)
		
	parsed_tree = build_tree(bckptr,delta,word_list,('S',0,N))
		
	return parsed_tree, delta['S'][0][N]


if __name__ == '__main__':

	sysarg_weights = sys.argv[1] #training file with weights
	sysarg_test= sys.argv[2] 

	weights_list = []
	with open(sysarg_weights) as weights_file:
		weights_list = [a_line for a_line in weights_file]

	grammar, symbols = read_weights_file(weights_list)

	with open(sysarg_test) as test_file:

		for a_line in test_file:
			print(a_line.strip())
			a_line = a_line.strip().split(" ")
			
			decoded_tree, weight = viterbi_cyk(a_line,grammar,symbols)
			print(weight, decoded_tree)
			print(" ")