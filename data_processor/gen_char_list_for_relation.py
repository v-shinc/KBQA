
def gen_char_list(fn_word, fn_sub_rel, fn_out):
	char_set = set()
	for fn in [fn_word, fn_sub_rel]:
		with open(fn) as fin:
			for line in fin:
				char_set.update(line.decode('utf8').strip())

	with open(fn_out, 'w') as fout:
		for c in char_set:
			print >> fout, c.encode('utf8')


if __name__ == '__main__':
	gen_char_list('../data/wq.simple.word.list.v3', '../data/wq.simple.sub.rel.list', '../data/relation.char.list')