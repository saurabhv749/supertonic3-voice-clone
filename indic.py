import re
from typing import Union

class HindiNumberPreprocessor:
    """
    A pre-processor class to convert numbers from Roman/Latin digits (0-9),
    classical Roman numerals (I, V, X...), or Devanagari digits (०-९) into Hindi words.
    """
    def __init__(self):
        # Precise vocabulary mapped directly from the Wikibooks reference
        self.hindi_nums = {
            0: "शून्य", 1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पांच", 6: "छः", 7: "सात", 8: "आठ", 9: "नौ", 10: "दश",
            11: "ग्यारह", 12: "बारह", 13: "तेरह", 14: "चौदह", 15: "पंद्रह", 16: "सोलह", 17: "सत्रह", 18: "अट्ठारह", 19: "उन्नीस", 20: "बीस",
            21: "इक्कीस", 22: "बाईस", 23: "तेईस", 24: "चौबिस", 25: "पच्चीस", 26: "छब्बीस", 27: "सत्ताईस", 28: "अट्ठाईस", 29: "उनतीस", 30: "तीस",
            31: "इकतीस", 32: "बत्तीस", 33: "तैंतीस", 34: "चौंतीस", 35: "पैंतीस", 36: "छत्तीस", 37: "सैंतीस", 38: "अड़तीस", 39: "उनतालीस", 40: "चालीस",
            41: "इकतालीस", 42: "बयालीस", 43: "तैंतालीस", 44: "चौंतालीस", 45: "पैंतालीस", 46: "छियालीस", 47: "सैंतालीस", 48: "अड़तालीस", 49: "उनचास", 50: "पचास",
            51: "इक्यावन", 52: "बावन", 53: "तिरपन", 54: "चौवन", 55: "पचपन", 56: "छप्पन", 57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ", 60: "साठ",
            61: "इकसठ", 62: "बासठ", 63: "तिरसठ", 64: "चौंसठ", 65: "पैंसठ", 66: "छाछठ", 67: "सड़सठ", 68: "अड़सठ", 69: "उनहत्तर", 70: "सत्तर",
            71: "इकहत्तर", 72: "बहत्तर", 73: "तिहत्तर", 74: "चौहत्तर", 75: "पचहत्तर", 76: "छिहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उन्यासी", 80: "अस्सी",
            81: "इक्यासी", 82: "बयासी", 83: "तिरासी", 84: "चौरासी", 85: "पचासी", 86: "छियासी", 87: "सत्तासी", 88: "अठासी", 89: "नवासी", 90: "नब्बे",
            91: "इक्यानवे", 92: "बानवे", 93: "तिरानवे", 94: "चौरानवे", 95: "पचानवे", 96: "छियानवे", 97: "सत्तानवे", 98: "अट्ठानवे", 99: "निन्यानवे", 100: "सौ"
        }

        # Mapping for Devanagari script digits to standard Latin script digits
        self.devanagari_to_latin = {
            '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
            '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
        }

        # Regex to catch classical Roman alphabet numerals (e.g., XIV, C)
        self.classical_roman_pattern = re.compile(r'^[IVXLCDM]+$', re.IGNORECASE)

    def _classical_roman_to_int_str(self, s: str) -> str:
        """Converts classical Roman numerals (e.g., XIV) into standard integer string representation."""
        roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        prev_value = 0
        for char in reversed(s.upper()):
            value = roman_map.get(char, 0)
            if value >= prev_value:
                total += value
            else:
                total -= value
            prev_value = value
        return str(total)

    def normalize_input(self, val: Union[str, int, float]) -> str:
        """Normalizes any input type into a clean string containing standard 0-9 digits and decimals."""
        if isinstance(val, (int, float)):
            s = str(val)
        elif isinstance(val, str):
            s = val.strip().replace(',', '')  # Strip spaces and formatting commas
        else:
            raise ValueError("Unsupported input format. Input must be a string, int, or float.")

        # Convert classical Roman numerals if encountered
        if self.classical_roman_pattern.match(s):
            s = self._classical_roman_to_int_str(s)

        # Standardize Devanagari numerals to standard digits
        normalized_chars = [self.devanagari_to_latin.get(char, char) for char in s]
        return "".join(normalized_chars)

    def _integer_to_hindi(self, num: int) -> str:
        """Processes the integer portion into Hindi phrase logic according to the Indian numbering rules."""
        if num == 0:
            return self.hindi_nums[0]

        parts = []

        # Lakhs (1,00,000)
        lakhs = num // 100000
        if lakhs > 0:
            lakh_word = self.hindi_nums.get(lakhs, str(lakhs))
            parts.append(f"{lakh_word} लाख")
            num %= 100000

        # Thousands (1,000)
        thousands = num // 1000
        if thousands > 0:
            thousand_word = self.hindi_nums.get(thousands, str(thousands))
            parts.append(f"{thousand_word} हज़ार")
            num %= 1000

        # Hundreds (100)
        hundreds = num // 100
        if hundreds > 0:
            hundred_word = self.hindi_nums.get(hundreds, str(hundreds))
            num %= 100
            if hundred_word == "एक" and num == 0:
                # we don't say 'एक सौ रुपये', just 'सौ रुपये' 
                parts.append(f"सौ")
            else:
                parts.append(f"{hundred_word} सौ")

        # Base remainder numbers (1-99)
        if num > 0:
            parts.append(self.hindi_nums[num])

        return " ".join(parts)


    def process_mixed_string(self, text: str) -> str:
        """Processes a string that may contain mixed numerical and non-numerical tokens."""
        words = text.split()
        processed_words = []

        for word in words:
            try:
                # First, check if it's a classical Roman numeral
                if self.classical_roman_pattern.match(word):
                    normalized_roman = self._classical_roman_to_int_str(word)
                    converted_word = self._integer_to_hindi(int(normalized_roman))
                    processed_words.append(converted_word)
                    continue

                # Normalize Devanagari digits to Latin digits and remove commas
                normalized_val = self.normalize_input(word)

                # Check if the normalized string is purely numeric (Latin digits, optional decimal)
                # and not just an empty string or a decimal point
                if re.fullmatch(r'^\d*(\.\d+)?$', normalized_val) and normalized_val not in ("", "."):
                    if '.' in normalized_val:
                        int_part_str, frac_part_str = normalized_val.split('.', 1)
                    else:
                        int_part_str, frac_part_str = normalized_val, ""

                    int_val = int(int_part_str) if int_part_str else 0
                    int_hindi = self._integer_to_hindi(int_val)

                    if frac_part_str:
                        frac_words = [self.hindi_nums[int(digit)] for digit in frac_part_str]
                        frac_hindi = "दशमलव " + " ".join(frac_words)
                        converted_word = f"{int_hindi} {frac_hindi}".strip()
                    else:
                        converted_word = int_hindi
                    processed_words.append(converted_word)
                else:
                    # If not a pure number or recognized Roman numeral, keep the original word
                    processed_words.append(word)
            except (ValueError, TypeError):
                processed_words.append(word) # Keep original word if it cannot be processed as a number
            except Exception: # Catch any other unexpected errors
                processed_words.append(word)

        return " ".join(processed_words)
