

# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])


def word_tokenize(line):
    return line.split()
