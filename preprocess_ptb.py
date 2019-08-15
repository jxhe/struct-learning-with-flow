#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018-01-19 Junxian He <junxianh2@gmail.com>
#
# Distributed under terms of the MIT license.
from __future__ import print_function

import os
import shutil
import argparse
from nlp_commons import wsj10
from nlp_commons.dep import dwsj

def generate_file(dir_name, fname, max_length=10):
    data_reader = dwsj.DepWSJ(max_length=max_length, basedir=dir_name)

    print('complete reading data')

    tag_sents, _ = data_reader.tagged_sents()
    deps_total = data_reader.get_gold_dep()
    with open(fname, "w") as fout:
        for tag_sent, sent_deps in zip(tag_sents, deps_total):
            deps = sent_deps.deps
            for i, (tag_word, dep) in enumerate(zip(tag_sent, deps)):
                fout.write('%d\t%s\t%s\t%d\n' % (i+1, tag_word[1], tag_word[0], dep[1]+1))
            fout.write('\n')

parser = argparse.ArgumentParser(description='preprocess ptb data')
parser.add_argument('--ptbdir', type=str, help='input directory')
# parser.add_argument('--task', type=str, choices=["tag", "parse"],
#     default="tag")

args = parser.parse_args()

if not os.path.exists("tmp_train"):
    os.makedirs("tmp_train")

if not os.path.exists("tmp_test"):
    os.makedirs("tmp_test")

abs_ptb = os.path.abspath(args.ptbdir)

for i in range(2, 22):
    ind = str("%02d" % i)
    if not os.path.exists("tmp_train/%02d" % i):
        os.symlink(os.path.join(abs_ptb, ind), "tmp_train/%02d" % i)

ind = 23
if not os.path.exists("tmp_test/%02d" % ind):
    os.symlink(os.path.join(abs_ptb, str(ind)), "tmp_test/%02d" % ind)

outdir = "ptb_parse_data"
if not os.path.exists(outdir):
    os.makedirs(outdir)

print("generate train file (len <= 10)")
generate_file("tmp_train", os.path.join(outdir, "ptb_parse_train_len10.txt"))

print("generate test file")
generate_file("tmp_test", os.path.join(outdir, "ptb_parse_test.txt"), max_length=200)

shutil.rmtree("tmp_train")
shutil.rmtree("tmp_test")
