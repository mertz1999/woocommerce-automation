import re

def fa_to_en(text):
    #convert english to persiann
    fa_digits = '۰۱۲۳۴۵۶۷۸۹'
    en_digits = '0123456789'
    trans_table = str.maketrans(fa_digits, en_digits)
    return text.translate(trans_table)

def extract_price_and_duration(query_text):
    query_text = fa_to_en(query_text)

    # extract price
    price_pattern = r'(\d[\d,\.]*)\s*(?:(هزار|میلیون|میلیارد)?\s*)?(تومان|ریال)'
    price_match = re.search(price_pattern, query_text)

    if price_match:
        raw_number = price_match.group(1).replace(',', '').replace('٬', '')
        unit = price_match.group(2)
        currency = price_match.group(3)

        price = float(raw_number)

        # convert tabaghe 
        multiplier = {
            None: 1,
            'هزار': 1_000,
            'میلیون': 1_000_000,
            'میلیارد': 1_000_000_000
        }.get(unit, 1)

        price *= multiplier

        # convert to rial
        if currency == 'تومان':
            price *= 10  # هر تومان = 10 ریال
        

        price = int(price)
    else:
        price = None

    # extract course time
    

    pattern_hour_minute = r'(\d+)\s*ساعت(?:\s*و\s*(\d+)\s*دقیقه)?'
    match = re.search(pattern_hour_minute, query_text)

    if match:  #hour must be and minutes can be or not
        hours = int(match.group(1))
        minutes = int(match.group(2)) if match.group(2) else 0
        duration = f' ساعت {hours}'
    else:
        # حالت 2: فقط دقیقه
        pattern_minute_only = r'(\d+)\s*دقیقه'
        match = re.search(pattern_minute_only, query_text)
        if match:
            hours = 0
            minutes = int(match.group(1))
            duration = f' دقیقه {minutes}'
        else:
            hours = 0
            minutes = 0
            duration= None


    return price, duration


if __name__ == "__main__":
 
  price , duration =extract_price_and_duration("آموزش فتوشاپ 4 ساعت")
  print(price)
  print(duration)