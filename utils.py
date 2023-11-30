import re
from word2number import w2n

MAX_N = 100
MIN_N = 10
MAX_M=200

numbers_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
                 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
                 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
                 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty',
                 'seventy', 'eighty', 'ninety', 'hundred', 'thousand',
                 'million']
numbers_pattern = r'|'.join(numbers_names)
and_pattern = r'\s+(?:and\s+)?'
digit_pattern = r'\d+(?:,\d{3})*(?:\.\d+)?'
pattern = fr'\b(?:(?:{numbers_pattern})(?:(?:{and_pattern})\b(?:{numbers_pattern}))*|{digit_pattern})\b'


def find_numbers_in_text(text):
    # Split the text into words
    text = text.lower()
    text = text.replace('-', ' ')
    text = text.replace('&', ' and ')
    text = text.replace('&&', ' and ')
    # Find all numbers in the text
    numbers = re.findall(pattern, text, flags=re.IGNORECASE)
    matches = []
    for num in numbers:
        num = num.replace(',', '')
        if len(num) > 100:
            continue
        try:
            num = int(num)
            matches.append(num)
            continue
        except:
            try:
                num = float(num)
                matches.append(num)
                continue
            except:
                try:
                    num = w2n.word_to_num(num)
                    matches.append(num)
                    continue
                except:
                    pass

    return matches



