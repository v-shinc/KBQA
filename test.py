def find_word(sentence, word):
    word_len = len(word)
    sentence_len = len(sentence)
    text = ' '.join(word)
    for i in range(sentence_len - word_len + 1):
        if sentence[i] == word[0] and ' '.join(sentence[i:i + word_len]) == text:
            return i
    return -1

print find_word([u'<START>', u'what', u'happened', u'after', u'mr', u'sugihara', u'died', u'<END>'], [u'mr', u'sugihara'])