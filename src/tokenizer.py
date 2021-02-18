import re

REGEX_SPECIAL_CHARACTERS = ['+', '-', '*', '|', '&', '[', ']', '(', ')', '{',
		'}', '^', '?', '.', '$', ',', ':', '=', '#', '!', '<']

class Tokenizer:
	''' Simple tokenizer, based on provided separators. The tokenizer splits
	input strings at positions where one or more separators occur. It
	returns the tokens as a list of string. The returned list of tokens does
	not contain any empty tokens ('' string).

	Example:
	* Separators: [' ', '.']
	* Input string: 'This is a test... of the tokenizer.'
	* Output tokens: ['This', 'is', 'a', 'test', 'of', 'the', 'tokenizer']
	'''


	def __init__(self, separators):
		''' Constructor for a Tokenizer object.

		The tokenizer splits input strings based on the provided
		separators.

		:param separators: Separators of the tokenizer
		:type separators: List or array of strings
		'''
		separators = ['\\'+sep if sep in REGEX_SPECIAL_CHARACTERS else sep for sep in separators]
		self._regex = '[' + ''.join(separators) + ']+'

	def tokenize(self, s):
		''' Tokenize a string

		Tokenize a string based on the separators of the tokenizer.

		:param s: The string to tokenize.
		:type s: String

		:return: List of tokens
		:rtype: List of String
		'''
		return [t for t in re.split(self._regex, s) if t != '']
