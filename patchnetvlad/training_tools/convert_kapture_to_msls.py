#!/usr/bin/env python

import sys

infilename = sys.argv[1]
outfilename = sys.argv[2]

query_ref_map = {}

with open(infilename, 'r') as infile:
    for idx, l in enumerate(infile):
        if l.startswith('#'):
            continue

        query_with_path, ref_with_path = l.split(',')
        query_img_name = query_with_path[query_with_path.rfind('/')+1:query_with_path.find('.jpg')]
        ref_img_name = ref_with_path[ref_with_path.rfind('/')+1:ref_with_path.find('.jpg')]

        if query_img_name not in query_ref_map:
            query_ref_map[query_img_name] = []
        query_ref_map[query_img_name].append(ref_img_name)

with open(outfilename, 'w') as outfile:
    for query_img_name in query_ref_map:
        outfile.write(query_img_name)
        outfile.write(' ')
        outfile.write(' '.join(query_ref_map[query_img_name]))
        outfile.write('\n')
