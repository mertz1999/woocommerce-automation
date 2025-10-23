import re
import db_init

def fa_to_en(text):

    #convert english to persiann
    fa_digits = '۰۱۲۳۴۵۶۷۸۹'
    en_digits = '0123456789'
    trans_table = str.maketrans(fa_digits, en_digits)
    return text.translate(trans_table)

def convert_time_standard(time_intext) 
   time_intext = time_intext.strip().lower()

    # نگاشت عددهای فارسی و حروفی به عدد
    num_words = {
        "یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "هفت": 7, "هشت": 8, "نه": 9, "ده": 10,
        "یازده": 11, "دوازده": 12, "سیزده": 13, "چهارده": 14, "پانزده": 15, "شانزده": 16,
        "هفده": 17, "هجده": 18, "نوزده": 19, "بیست": 20, "سی": 30, "چهل": 40, "پنجاه": 50, "شصت": 60
    }

    # تابع کمکی برای تبدیل عدد به عدد صحیح (چه حروفی چه رقمی)
    def to_number(s):
        s = s.strip()
        if s.isdigit():
            return int(s)
        return num_words.get(s, 0)

    total_minutes = 0

    # ۱. اگر ورودی فقط دقیقه باشد
    if "دقیقه" in time_intext :
        match = re.search(r"(\d+)", time_intext)
        if match:
            return int(match.group(1))
        # عدد حروفی
        for k, v in num_words.items():
            if k in text:
                return v

    # ۲. اگر ساعت و دقیقه باشد
    hour_match = re.search(r"(\d+|\D+?)\s*ساعت", time_intext)
    minute_match = re.search(r"(\d+|\D+?)\s*دقیقه", time_intextt)

    hours = 0
    minutes = 0

    if hour_match:
        hours = to_number(hour_match.group(1))
    if minute_match:
        minutes = to_number(minute_match.group(1))

    if hours or minutes:
        total_minutes = hours * 60 + minutes
        return total_minutes

    # ۳. حالت فقط ساعت (مثل "حدود دو ساعت")
    if "ساعت" in time_intext:
        for k, v in num_words.items():
            if k in text:
                return v * 60
        match = re.search(r"(\d+)", time_intext)
        if match:
            return int(match.group(1)) * 60

    # ۴. حالت فقط عدد (مثلاً "120")
    if time_intext.isdigit():
        return int(time_intext)

    return 0

def convert_price_standard(price_intext)
   price_pattern = r'(\d[\d,\.]*)\s*(?:(هزار|میلیون|میلیارد)?\s*)?(تومان|ریال)'
   price_match = re.search(price_pattern, price_intext)
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
    return price

def sql_functions():
    
 conn = db_init.get_conn()
 cursor = conn.cursor()

 sql_function_time = """
   CREATE OR REPLACE FUNCTION normalize_duration(duration_text TEXT)
   RETURNS INTEGER AS $$
   DECLARE
    hours INTEGER := 0;
    minutes INTEGER := 0;
   BEGIN
    IF duration_text ~ 'ساعت' THEN
        hours := COALESCE((regexp_replace(duration_text, '.*?([0-9]+)\s*ساعت.*', '\\1')::INTEGER), 0);
    END IF;
    IF duration_text ~ 'دقیقه' THEN
        minutes := COALESCE((regexp_replace(duration_text, '.*?([0-9]+)\s*دقیقه.*', '\\1')::INTEGER), 0);
    END IF;
    RETURN (hours * 60 + minutes);
   END;
   $$ LANGUAGE plpgsql IMMUTABLE """

  cursor.execute(sql_function_time)
  conn.commit()

 sql_function_price="""  CREATE OR REPLACE FUNCTION to_standard_rial(input_text TEXT)
    RETURNS BIGINT AS $$
    DECLARE
    cleaned TEXT;
    num BIGINT := 0;
    is_toman BOOLEAN := FALSE;
    is_rial  BOOLEAN := FALSE;
    multiplier BIGINT := 1;
    BEGIN

     cleaned := TRANSLATE(input_text, '۰۱۲۳۴۵۶۷۸۹', '0123456789');
     IF cleaned ~ '^[0-9]+$' THEN
        RETURN cleaned::BIGINT;
     END IF;

    -- تشخیص واحد
     IF input_text ILIKE '%تومان%' THEN
        is_toman := TRUE;
     ELSIF input_text ILIKE '%ریال%' THEN
        is_rial := TRUE;
     END IF;

     -- حذف فاصله‌ها و کلمات واحد
     cleaned := REPLACE(REPLACE(REPLACE(input_text, 'تومان', ''), 'ریال', ''), ' ', '');

     -- تبدیل اعداد فارسی به انگلیسی
     cleaned := TRANSLATE(cleaned, '۰۱۲۳۴۵۶۷۸۹', '0123456789');

     -- بررسی مقیاس‌ها
     IF cleaned ~ 'هزار' THEN
        cleaned := REPLACE(cleaned, 'هزار', '');
        num := COALESCE(cleaned::BIGINT, 0) * 1000;
     ELSIF cleaned ~ 'میلیون' THEN
        cleaned := REPLACE(cleaned, 'میلیون', '');
        num := COALESCE(cleaned::BIGINT, 0) * 1000000;
     ELSIF cleaned ~ 'میلیارد' THEN
        cleaned := REPLACE(cleaned, 'میلیارد', '');
        num := COALESCE(cleaned::BIGINT, 0) * 1000000000;
   
       
     END IF;

     -- تعیین ضریب تبدیل
     IF is_toman THEN
        multiplier := 10;  
     ELSE
        multiplier := 1;   
     END IF;

     RETURN num * multiplier;
    END;
    $$ LANGUAGE plpgsql;
     """
  cursor.execute(sql_function_time)
  conn.commit()

def extract_normal_filters(**kwargs):# dictionary may be in diffrent format so must be declare each key(filter) has which kind(price,time, ...)

 for key, value in kwargs.items():
    if value == "price":
       
       sql_query = f"""
        UPDATE products
        SET price = to_standard_rial(price)::TEXT;
        """
        cursor.execute(sql_query)
        conn.commit()

    elif value == "time":

       sql_query = f"""
        ALTER TABLE products
        ADD COLUMN %s integer GENERATED ALWAYS AS (
            normalize_duration(
                (
                     SELECT 
                     jsonb_path_query_first(metadata, '$.**.%s') AS first_target
                     FROM products

                )
            )
        ) STORED;
        """
        cursor.execute(sql_query,key,key)
        conn.commit()
   
    elif value == "date":
        sql_query = f"""
        ALTER TABLE products
        ADD COLUMN %s integer GENERATED ALWAYS AS (
            normalize_date(   #function must be write in top
                (
                     SELECT 
                     jsonb_path_query_first(metadata, '$.**.%s') AS first_target
                     FROM products

                )
            )
        ) STORED;
        """
        cursor.execute(sql_query,key, key)
        conn.commit()
      
    elif value== "size":
        sql_query = f"""
        ALTER TABLE products
        ADD COLUMN %s integer GENERATED ALWAYS AS (
            normalize_size(   #function must be write in top
                (
                     SELECT 
                     jsonb_path_query_first(metadata, '$.**.%s') AS first_target
                     FROM products

                )
            )
        ) STORED;
        """
        cursor.execute(sql_query,key,key)
        conn.commit()

    else: # for text filter no need to run function on filter
        sql_query = f"""
        ALTER TABLE products
        ADD COLUMN %s integer GENERATED ALWAYS AS (
                    (
                     SELECT 
                     jsonb_path_query_first(metadata, '$.**.%s') AS first_target
                     FROM products

                )
            
        ) STORED;
        """
        cursor.execute(sql_query,key,key)
        conn.commit()



 



