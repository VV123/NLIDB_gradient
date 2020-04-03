# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import sys
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path).replace('end2end_anno', 'annotation'))
import json
from argparse import ArgumentParser
from tqdm import tqdm
from lib.query import Query
from lib.common import count_lines
import numpy as np
import re
import collections
from collections import defaultdict
import copy
import editdistance as ed
import scipy
from scipy import spatial
from utils import _truncate, _clean
from utils import _preclean
save_path = os.path.dirname(path).replace('utils/end2end_anno',
                                          'data/end2end')
data_path = os.environ['WIKI_PATH']
maps = defaultdict(list)
stop_words = ['a', 'of', 'the', 'in']
UNK = 'unk'

def _gen_files(save_path=save_path):
    """
    Prepare files used for Machine Comprehension Binary Classifier
    """
    for split in ['train', 'test', 'dev']:
        print('------%s-------'%split)
        n = 0
        fsplit = os.path.join(data_path, split) + '.jsonl'
        ftable = os.path.join(data_path, split) + '.tables.jsonl'
        """
	test.txt original column content w/o truncate
	test_model.txt column name truncate or pad to length 3
	"""
        with open(fsplit) as fs, open(ftable) as ft, \
            open(os.path.join(save_path, split+'.txt'), mode='w') as fw, \
            open(os.path.join(save_path, '%s_model.txt'%split), mode='w') as fw_model, \
			open(os.path.join(save_path, split+'.lon'), mode='w') as fsql, \
            open(os.path.join(save_path, '%s.ori.qu'%split), 'w') as qu_file, \
            open(os.path.join(save_path, '%s.ori.lon'%split), 'w') as lon_file:
            print('loading tables...')
            tables = {}
            for line in tqdm(ft, total=count_lines(ftable)):
                d = json.loads(line)
                tables[d['id']] = d
            print('loading tables done.')

            print('loading examples')
            f2v_all, v2f_all = [], []
            for line in tqdm(fs, total=count_lines(fsplit)):

                d = json.loads(line)
                Q = d['question']
                Q = _preclean(Q).replace('\t','')
				
                qu_file.write(Q+'\n')

                q_sent = Query.from_dict(d['sql'])
                rows = tables[d['table_id']]['rows']
                S, col_names, val_names = q_sent.to_sentence(
                    tables[d['table_id']]['header'], rows,
                    tables[d['table_id']]['types'])
                S = _preclean(S)
				
                lon_file.write(S+'\n')

                rows = np.asarray(rows)
                fs = tables[d['table_id']]['header']
                all_fields = [ _preclean(f) for f in fs]
                # all fields are sorted by length in descending order
                # for string match purpose
                headers = sorted(all_fields, key=len, reverse=True)

                f2v = defaultdict(list)  #f2v
                v2f = defaultdict(list)  #v2f
                for row in rows:
                    for i in range(len(fs)):
                        cur_f = _preclean(str(fs[i]))
                        cur_row = _preclean(str(row[i]))
                        #cur_f = cur_f.replace('\u2003',' ')
                        f2v[cur_f].append(cur_row)
                        if cur_f not in v2f[cur_row]:
                            v2f[cur_row].append(cur_f)
                f2v_all.append(f2v)
                v2f_all.append(v2f)

                #####################################
                ########## Annotate SQL #############
                #####################################
                q_sent = Query.from_dict(d['sql'])
                S, col_names, val_names = q_sent.to_sentence(
                    tables[d['table_id']]['header'], rows,
                    tables[d['table_id']]['types'])
                S = _preclean(S)

                S_noparen = q_sent.to_sentence_noparenthesis(
                    tables[d['table_id']]['header'], rows,
                    tables[d['table_id']]['types'])
                S_noparen = _preclean(S_noparen)

                col_names = [ _preclean(col_name) for col_name in col_names ]
                val_names = [ _preclean(val_name) for val_name in val_names ]


                HEAD = col_names[-1]
                S_head = _preclean(HEAD)


                #annotate for SQL
                name_pairs = []
                for col_name, val_name in zip(col_names, val_names):
                    if col_name == val_name:
                        name_pairs.append([_preclean(col_name), 'true'])
                    else:
                        name_pairs.append(
                            [_preclean(col_name),
                             _preclean(val_name)])

                # sort to compare with candidates
                name_pairs.sort(key=lambda x: x[1])
                fsql.write('#%d\n'%n)
                fw.write('#%d\n'%n)

                for f in col_names:
                    fsql.write(S.replace(f,'['+f+']')+'\n')
                    f = _truncate(f, END ='<bos>', PAD = '<pad>', max_len = -1)
                    s = (Q + '\t' + f + '\t 1')
                    assert len(s.split('\t')) == 3
                    fw.write(s + '\n')
                 
                for f in [f for f in headers if f not in col_names]:
                    f = _truncate(f, END ='<bos>', PAD = '<pad>', max_len = -1)
                    s = (Q + '\t' + f + '\t 0')
                    assert len(s.split('\t')) == 3
                    fw.write(s + '\n')
                    fsql.write(S+'\n')

                #if '\u2003' in Q:
                #    print('u2003: '+Q)
                #if '\xa0' in Q:
                #    print(n)
                #    print('xa0: '+Q)
                #    print(S)
                for f in col_names:
                    f = f.replace(u'\xa0', u' ').replace('\t','')
                    Q = Q.replace(u'\xa0', u' ').replace('\t','')
                    f = _truncate(f, END ='bos', PAD = 'pad', max_len = 3)
                    s = (Q + '\t' + f + '\t 1')
                    assert len(s.split('\t')) == 3
                    fw_model.write(s + '\n')

                for f in [f for f in headers if f not in col_names]:
                    f = f.replace(u'\xa0', u' ').replace('\t','')
                    Q = Q.replace(u'\xa0', u' ').replace('\t','')
                    f = _truncate(f, END ='bos', PAD = 'pad', max_len = 3)
                    s = (Q + '\t' + f + '\t 0')
                    assert len(s.split('\t')) == 3
                    fw_model.write(s + '\n')
	
                n += 1
            fsql.write('#%d\n'%n)
            fw.write('#%d\n'%n)

        scipy.savez(os.path.join(save_path,'%s_dict.npz'%split), f2v_all=f2v_all, v2f_all=v2f_all)
        print('num of records:%d'%n)

if __name__ == '__main__':
    _gen_files()
