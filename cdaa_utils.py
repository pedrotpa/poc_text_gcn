import os
import json
import pandas as pd


def get_collection_info(data_dir):
    retv=[]
    with open(os.path.join(data_dir,"collection-info.json"), 'r') as f:
        retv = [{k: attrib[k] for k in ('problem-name', 'language')} for attrib in json.load(f) ]
    return retv

def get_problem_info(problem_dir): 
    with open(problem_dir+os.sep+"problem-info.json", 'r') as f:
        fj = json.load(f)
    return [attrib['author-name'] for attrib in fj['candidate-authors']], problem_dir+os.sep+fj['unknown-folder']

def get_ground_truth_info(problem_dir):
    with open(problem_dir+os.sep+"ground-truth.json", 'r') as f:
        fj = json.load(f)
    return [(attrib['unknown-text'], attrib['true-author']) for attrib in fj['ground_truth']]

def get_infos(data_dir):
    from collections import namedtuple
    Collection = namedtuple('Collection',['info','problem_list','path'])
    Problem = namedtuple('Problem',['name','path','candidates_list','uk_folder','uk_info','language'])
    collection_info = get_collection_info(data_dir)
    infos = Collection(collection_info,[],data_dir)

    for problem in collection_info:
        problem_dir = data_dir+os.sep+problem['problem-name']
        candidates_list, uk_folder = get_problem_info(problem_dir)
        uk_info = get_ground_truth_info(problem_dir)
        infos.problem_list.append(Problem(problem['problem-name'], problem_dir, candidates_list, uk_folder,uk_info, problem['language']))
    return infos

def read_file(f_path):
    import codecs
    with codecs.open(f_path,'r',encoding='utf-8') as f:
        return f.read()


def read_files_folder(path):
    import glob
    import codecs
    files = sorted(glob.glob(path+os.sep+'*.txt'))
    texts=[]
    for v in files:
        texts.append(read_file(v))
    return texts

def load_problem_data(problem):
    import itertools
    l = lambda candidate: (read_files_folder(problem.path + os.sep + candidate), candidate)
    train_df = pd.DataFrame(map(lambda l: list(itertools.chain(*l)), zip(*[(files,[candidate]*len(files)) for files,candidate in map( l ,problem.candidates_list)])))
    train_df = train_df.transpose()
    train_df.columns=['x','y']

    test_df = pd.DataFrame([[read_file(problem.uk_folder + os.sep + file_name) for file_name,_ in problem.uk_info], [gt for _, gt in problem.uk_info]])
    test_df = test_df.transpose()
    test_df.columns=['x','y']

    return train_df,test_df

def pd_load_problem_data(problem):
    train_df, test_df = load_problem_data(problem)

    train_idx = list(range(len(train_df)))
    train_labels = train_df['y'][train_idx]

    test_idx = test_df[test_df['y'] != "<UNK>"].index
    test_labels = test_df['y'][test_idx]
    test_idx = test_idx + len(train_df)
    
    return pd.concat([train_df, test_df],keys=['train','test']), train_idx, test_idx, train_labels, test_labels
