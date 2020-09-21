#!/usr/bin/python3

import numpy as np
import itertools
from pyparsing import nestedExpr
import sys

def build_tree(bckptr,delta,word_list,init):
    left_node,right_node,split_point = bckptr[init[0]][init[1]][init[2]]
    tree_string_snippet = " ("+str(init[0]) #recursively build this up
    
    # left branch
    if split_point-init[1]==1: #reached leaf node
        node_str = " ("+str(left_node)+" "+str(word_list[init[1]])+")"
        tree_string_snippet += node_str
    else:
        tree_string_snippet += build_tree(bckptr,delta,word_list,(left_node,init[1],split_point))
    
    # right branch
    if init[2]-split_point==1: #reached leaf node
        node_str = " ("+str(right_node)+" "+str(word_list[init[2]-1])+")"
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
        
    return parsed_tree.strip(), delta['S'][0][N]

class Tree(object):
    def __init__(self,x):
        self.left = None
        self.right = None
        self.data = x

def parse_treestringlist(tree_string_list):
    if len(tree_string_list)==2: #leaf node
        leaf_node = Tree((tree_string_list[0],tree_string_list[1]))
        return leaf_node
    
    nonleaf_node = Tree(tree_string_list[0])
    nonleaf_node.left = parse_treestringlist(tree_string_list[1])
    nonleaf_node.right = parse_treestringlist(tree_string_list[2])
    return nonleaf_node

def get_leaves(tree):
    leaf_set = []
    if tree.left==None: # this is leaf node
        leaf_set.extend(tree.data[1])
    else:
        leaf_set.extend(get_leaves(tree.left))
        leaf_set.extend(get_leaves(tree.right))
    return leaf_set

def get_preterminals(tree):
    preterminal_set = []
    if tree.left==None: # this is leaf node
        preterminal_set.extend(tree.data[0])
    else:
        preterminal_set.extend(get_preterminals(tree.left))
        preterminal_set.extend(get_preterminals(tree.right))
    return preterminal_set

def get_nonterminals(tree):
    nonterminal_set = []
    if tree.left:
        nonterminal_set.extend(tree.data)
        nonterminal_set.extend(get_nonterminals(tree.left))
        nonterminal_set.extend(get_nonterminals(tree.right))
    return nonterminal_set

def process_train_datapoint(train_string):
    symbols = {'pre_terminals' : set(),
              'non_terminals' : set(),
              'terminals' : set()}
    p = nestedExpr('(',')').parseString(train_string).asList()[0]
    tree = parse_treestringlist(p) 
    
    #generate terminals
    leaves= get_leaves(tree)
    symbols['terminals']= symbols['terminals'].union(set(leaves))
    
    #generate pre-terminals
    preterminals= get_preterminals(tree)
    symbols['pre_terminals']= symbols['pre_terminals'].union(set(preterminals))
    
    # generate non-terminals
    nonterminals= get_nonterminals(tree)
    symbols['non_terminals']= symbols['non_terminals'].union(set(nonterminals))
    
    return leaves,symbols,tree

def compare_trees(tree1,tree2):
    if (tree1 == None and tree2 == None) : 
        return 1
    elif (tree1 != None and tree2 == None) : 
        return 0
    elif (tree1 == None and tree2 != None) : 
        return 0
    else:
        if (tree1.data == tree2.data and 
            compare_trees(tree1.left, tree2.left)  
            and compare_trees(tree1.right, tree2.right)) : 
            return 1
        else: 
            return 0

def get_R_rules(tree):
    updates=[]
    if tree.left: # this node has children
        RHS0=tree.left.data if tree.left.left else tree.left.data[0]
        RHS1=tree.right.data if tree.right.left else tree.right.data[0]
        updates.append((tree.data,RHS0,RHS1))
        updates.extend(get_R_rules(tree.left))
        updates.extend(get_R_rules(tree.right))
    return set(updates)

def find_first_terminal(tree):
    if tree.left==None: #reached a leaf
        return tree.data[1]
    else:
        return find_first_terminal(tree.left)

def get_F_rules(tree):
    rules=[]
    if tree.left: # this node has children
        rules.append((tree.data,find_first_terminal(tree)))
        rules.extend(get_F_rules(tree.left))
        rules.extend(get_F_rules(tree.right))
    return set(rules)

def find_last_terminal(tree):
    if tree.right==None: #reached a leaf
        return tree.data[1]
    else:
        return find_last_terminal(tree.right)

def get_L_rules(tree):
    rules=[]
    if tree.right: # this node has children
        rules.append((tree.data,find_last_terminal(tree)))
        rules.extend(get_L_rules(tree.left))
        rules.extend(get_L_rules(tree.right))
    return set(rules)

def get_T_rules(tree):
    rules=[]
    if not tree.left: # this is leaf node
        rules.append((tree.data))
    else:    
        rules.extend(get_T_rules(tree.left))
        rules.extend(get_T_rules(tree.right))
    return set(rules)

if __name__ == '__main__':
	sysarg_train = sys.argv[1] #training file with weights
	sysarg_weightstrained= sys.argv[2] 

	train_list = []
	with open(sysarg_train) as train_file:
	    train_list = [a_line for a_line in train_file]

	# the program will try to learn this grammar dictionary from data    
	grammar = {'T':{},
	           'R':{},
	           'F':{},
	           'L':{}}

	# the program will enrich the symbols dict as it comes across different
	# symbols in the training datapoints
	symbols = {'pre_terminals' : set(),
	          'non_terminals' : set(),
	          'terminals' : set()}

	for iter_ in range(15):
	    print("iter: "+str(iter_))
	    err = 0
	    for a_line in train_list:
	        a_line = a_line.strip()
	        print("gold tree: "+a_line)
	        current_string, current_symbols,training_tree = process_train_datapoint(a_line)

	        #grow repository of symbols
	        symbols['pre_terminals'] = symbols['pre_terminals'].union(current_symbols['pre_terminals'])
	        symbols['non_terminals'] = symbols['non_terminals'].union(current_symbols['non_terminals'])
	        symbols['terminals'] = symbols['terminals'].union(current_symbols['terminals'])

	        #get viterbi cyk parsing
	        decoded_tree_string, weight = viterbi_cyk(current_string,grammar,symbols)

	        print("viterbi parse:")
	        print(weight,decoded_tree_string)

	        #generate proper tree for viterbi decoded tree-string
	        p = nestedExpr('(',')').parseString(decoded_tree_string).asList()[0]
	        decoded_tree = parse_treestringlist(p) 

	        #compare training and decoded trees
	        if compare_trees(training_tree,decoded_tree):
	            print("correct parse")
	        else: #update weights
	            print("incorrect parse")
	            err+=1
	            print("updated weights:")
	            
	            #update F weights
	            rules = get_F_rules(training_tree)
	            for rule in rules:
	                if (rule[0],rule[1]) not in grammar['F']:
	                    grammar['F'][(rule[0],rule[1])] = 0
	                grammar['F'][(rule[0],rule[1])] += 1
	                print("F_"+str(rule[0])+"_"+str(rule[1])+" "+str(grammar['F'][(rule[0],rule[1])]))

	            rules = get_F_rules(decoded_tree)
	            for rule in rules:
	                if (rule[0],rule[1]) not in grammar['F']:
	                    grammar['F'][(rule[0],rule[1])] = 0
	                grammar['F'][(rule[0],rule[1])] -= 1
	                print("F_"+str(rule[0])+"_"+str(rule[1])+" "+str(grammar['F'][(rule[0],rule[1])]))

	            #update L weights
	            rules = get_L_rules(training_tree)
	            for rule in rules:
	                if (rule[0],rule[1]) not in grammar['L']:
	                    grammar['L'][(rule[0],rule[1])] = 0
	                grammar['L'][(rule[0],rule[1])] += 1
	                print("L_"+str(rule[0])+"_"+str(rule[1])+" "+str(grammar['L'][(rule[0],rule[1])]))

	            rules = get_L_rules(decoded_tree)
	            for rule in rules:
	                if (rule[0],rule[1]) not in grammar['L']:
	                    grammar['L'][(rule[0],rule[1])] = 0
	                grammar['L'][(rule[0],rule[1])] -= 1
	                print("L_"+str(rule[0])+"_"+str(rule[1])+" "+str(grammar['L'][(rule[0],rule[1])]))
	            
	            # update R weights
	            rules = get_R_rules(training_tree)
	            for rule in rules:
	                if (rule[0],rule[1],rule[2]) not in grammar['R']:
	                    grammar['R'][(rule[0],rule[1],rule[2])] = 0
	                grammar['R'][(rule[0],rule[1],rule[2])] += 1
	                print("R_"+str(rule[0])+"_"+str(rule[1])+"_"+str(rule[2])+" "+str(grammar['R'][(rule[0],rule[1],rule[2])]))

	            rules = get_R_rules(decoded_tree)
	            for rule in rules:
	                if (rule[0],rule[1],rule[2]) not in grammar['R']:
	                    grammar['R'][(rule[0],rule[1],rule[2])] = 0
	                grammar['R'][(rule[0],rule[1],rule[2])] -= 1
	                print("R_"+str(rule[0])+"_"+str(rule[1])+"_"+str(rule[2])+" "+str(grammar['R'][(rule[0],rule[1],rule[2])]))
	            
	            # update T rules
	            rules = get_T_rules(training_tree)
	            for rule in rules:
	                if (rule[0],rule[1]) not in grammar['T']:
	                    grammar['T'][(rule[0],rule[1])] = 0
	                grammar['T'][(rule[0],rule[1])] += 1
	                print("T_"+str(rule[0])+"_"+str(rule[1])+" "+str(grammar['T'][(rule[0],rule[1])]))
	            
	            rules = get_T_rules(decoded_tree)
	            for rule in rules:
	                if (rule[0],rule[1]) not in grammar['T']:
	                    grammar['T'][(rule[0],rule[1])] = 0
	                grammar['T'][(rule[0],rule[1])] -= 1
	                print("T_"+str(rule[0])+"_"+str(rule[1])+" "+str(grammar['T'][(rule[0],rule[1])]))
	            
	        print(" ")
	    
	    print("errors in this iter: "+str(err))
	    print(" ")
	    
	    if not err:
	        print("No errors in this iter, stopping now")
	        break

	print(" ")
	print("Writing to disk")
	f= open(sysarg_weightstrained,"w")
	for key,value in grammar['F'].items():
	    f.write("F_"+key[0]+"_"+key[1]+" "+str(value)+"\r\n")
	for key,value in grammar['L'].items():
	    f.write("L_"+key[0]+"_"+key[1]+" "+str(value)+"\r\n")
	for key,value in grammar['R'].items():
	    f.write("R_"+key[0]+"_"+key[1]+"_"+key[2]+" "+str(value)+"\r\n")
	for key,value in grammar['T'].items():
	    f.write("T_"+key[0]+"_"+key[1]+" "+str(value)+"\r\n")
	f.close()
	    