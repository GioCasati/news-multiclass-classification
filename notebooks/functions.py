import html
import re
import unicodedata
import numpy as np
import pandas as pd


class TextCleaner:
    """
    This class is a configurable text preprocessing pipeline designed to convert raw text into a clean, normalized
    format suitable for Natural Language Processing.vIt works by passing a string through a series of "filters"
    (regular expressions and string manipulations) based on boolean flags set during initialization.
    """
    RE_HTML_METADATA = re.compile(r'<[^>]*(?:alt|title)=["\']([^"\']+)["\'][^>]*>')
    RE_A_TAG_HREF = re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>')  #new
    RE_HTML_TAGS = re.compile(r'<[^>]+>')
    RE_URL = re.compile(r'(https?://\S+|www\.\S+)', re.IGNORECASE)
    RE_EMAIL = re.compile(r'\S+@\S+')
    RE_CURRENCY = re.compile(r'[$€£]+')
    RE_NON_ALPHANUM = re.compile(r"[^a-zA-Z0-9\s]")
    RE_NUMBERS = re.compile(r'[0-9]+')
    RE_SHORT_WORDS = re.compile(r'\b[^\d\s]{1,2}\b')
    RE_WHITESPACE = re.compile(r'\s+')
    RE_URL_SEPARATORS = re.compile(r'[^a-zA-Z0-9]+')
    url_stopwords = {
        'http', 'https', 'www', 'com', 'org', 'net', 'html', 'htm',
        'php', 'aspx', 'jsp', 'rss', 'xml'
    }

    def __init__(self,
                 extract_html_metadata=True,
                 extract_url_text=True,
                 html_strip=True,
                 remove_urls=True,
                 remove_emails=True,
                 convert_currency=True,
                 normalize_unicode=True,
                 remove_non_alphanum=True,
                 remove_numbers=True,
                 lower=True,
                 remove_short=True):
        """
        Initializes the TextCleaner with specific cleaning rules.

        :param extract_html_metadata: If True, replaces HTML tags containing 'alt' or
                                          'title' attributes with the text content of those attributes.
        :param extract_url_text: If True, extracts meaningful words from links (both <a> tags
                                     and raw URLs) instead of removing them entirely.
        :param html_strip: If True, removes all HTML tags and unescapes HTML entities.
        :param remove_urls: If True, detects URLs. If `extract_url_text` is also True,
                                converts them to words; otherwise, deletes them.
        :param remove_emails: If True, removes email addresses.
        :param convert_currency: If True, replaces currency symbols ($, €, £) with the
                                     placeholder token 'moneytoken'.
        :param normalize_unicode: If True, performs NFKD normalization and removes non-ASCII
                                      characters (e.g., accents, emojis).
        :param remove_non_alphanum: If True, replaces non-alphanumeric characters with spaces.
        :param remove_numbers:  If True, removes all digits.
        :param lower: If True, converts text to lowercase.
        :param remove_short: If True, removes words fewer than 3 characters long.
        """
        self.extract_html_metadata = extract_html_metadata
        self.extract_url_text = extract_url_text
        self.html_strip = html_strip
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.convert_currency = convert_currency
        self.normalize_unicode = normalize_unicode
        self.remove_non_alphanum = remove_non_alphanum
        self.remove_numbers = remove_numbers
        self.lower = lower
        self.remove_short = remove_short

    def _url_to_words(self, match):
        """
        Internal helper to convert a URL regex match into a string of keywords.
        Splits the URL by non-alphanumeric characters, filters out common
        URL stopwords (com, www, html), and removes short segments.

        :param match: The regex match object containing the URL group.
        :return: A space-separated string of meaningful words extracted from the URL.
        """
        url = match.group(1)
        tokens = self.RE_URL_SEPARATORS.split(url)
        clean_tokens = []
        for t in tokens:
            t_lower = t.lower()
            if (len(t) > 2 and
                    not t.isdigit() and
                    t_lower not in self.url_stopwords):
                clean_tokens.append(t)
        return " " + " ".join(clean_tokens) + " "

    def clean_text(self, text):
        """
        Executes the cleaning pipeline on the input text, performing the steps
        specified at the construction following this order:
        1. Extract HTML metadata (alt/title).
        2. Extract words from <a> tag hrefs.
        3. Strip HTML tags and entities.
        4. Process raw URLs (remove or convert).
        5. Remove emails.
        6. Tokenize currency.
        7. Normalize Unicode (to ASCII).
        8. Remove non-alphanumeric characters.
        9. Remove numbers.
        10. Lowercase text.
        11. Remove short words (<= 2 chars).
        12. Collapse whitespace.

        :param text: The raw input string.
        :return: The cleaned and normalized string. Returns empty string if input is not a string.
        """
        if not isinstance(text, str):
            return ''
        if self.extract_html_metadata:
            text = self.RE_HTML_METADATA.sub(r' \1 ', text)
        if self.extract_url_text:
            text = self.RE_A_TAG_HREF.sub(self._url_to_words, text)
        if self.html_strip:
            text = self.RE_HTML_TAGS.sub(' ', text)
            text = html.unescape(text)
        if self.remove_urls:
            if self.extract_url_text:
                text = re.sub(r'(https?://\S+|www\.\S+)',
                              lambda m: self._url_to_words(re.match(r'(.*)', m.group(0))),
                              text, flags=re.IGNORECASE)
            else:
                text = self.RE_URL.sub(' ', text)
        if self.remove_emails:
            text = self.RE_EMAIL.sub(' ', text)
        if self.convert_currency:
            text = self.RE_CURRENCY.sub(' moneytoken ', text)
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        if self.remove_non_alphanum:
            text = self.RE_NON_ALPHANUM.sub(' ', text)
        if self.remove_numbers:
            text = self.RE_NUMBERS.sub(' ', text)
        if self.lower:
            text = text.lower()
        if self.remove_short:
            text = self.RE_SHORT_WORDS.sub(' ', text)
        text = self.RE_WHITESPACE.sub(' ', text).strip()
        return text

def global_cleaner(df: pd.DataFrame, cleaning_params=None):
    """
    Performs high-level data cleaning on a DataFrame and applies text normalization
    to specific content columns using TextCleaner. This function handles common database
    dump artifacts (like '\\N' or zero timestamps), cleans the 'source' metadata column,
    and applies the TextCleaner pipeline to  the 'title' and 'article' columns.

    :param df: The input DataFrame.
    :param cleaning_params: A dictionary of parameters to pass to the TextCleaner constructor.
    :return: The cleaned DataFrame with normalized text and handled missing values.
    """
    df = df.replace(to_replace=r'\\N', value=np.nan, regex=True)
    df = df.replace(to_replace='0000-00-00 00:00:00', value=np.nan)
    df['source'] = df['source'].astype(str).str.replace(r'\\', '', regex=True)
    df['source'] = df['source'].replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
    if cleaning_params is not None:
        cleaner = TextCleaner(**cleaning_params)
    else:
        cleaner = TextCleaner()
    df['title'] = df['title'].apply(cleaner.clean_text)
    df['article'] = df['article'].apply(cleaner.clean_text)
    return df

def deduplicate(df: pd.DataFrame, duplicate_keys, target_name):
    """
    The function solves conflicts in a group of identical rows that present different labels.
    In case of a majority, the most common label is preserved.
    In case of balanced labeling (tie), the whole group is dropped.

    :param df: Original dataset.
    :param duplicate_keys: Relevant columns' names for duplication check.
    :param target_name: Target column's name.
    :return: A cleaned version of the original dataset.
    """
    def solve_duplicates(duplicates):
        label_counts = duplicates[target_name].value_counts()
        if len(label_counts) == 1:
            return duplicates.iloc[[0]]
        first = label_counts.iloc[0]
        second = label_counts.iloc[1]
        if first == second:
            return duplicates.iloc[0:0]
        majority_label = label_counts.idxmax()
        return duplicates[duplicates[target_name] == majority_label].iloc[[0]]
    rows_to_keep = (
        df
        .groupby(duplicate_keys, group_keys=False)
        .apply(solve_duplicates, include_groups=False)
        .index
    )
    return df.loc[rows_to_keep]

def process_text_columns(df: pd.DataFrame, text_length=True, title_length=True,
                         article_length=True, digit_density=True):
    """
    The function joins the article and title columns in a single column named 'full_text', and extracts some metadata
    from both the joined and the original versions. It then processes the text to make it suitable for tf-idf
    representation, removing eventual irregularities left for the metadata extraction from the first cleaning step.

    :param df: DataFrame made of the 'title' and 'article' columns of the dataset.
    :param text_length: Whether to add the 'text_length' column or not.
    :param title_length: Whether to add the 'title_length' column or not.
    :param article_length: Whether to add the 'article_length' column or not.
    :param digit_density: Whether to add the 'digit_density' column or not.
    :return: A DataFrame made by the 'full_text' columns and the other ones specified by the parameters.
    """
    df_dict = dict()
    texts = df['title'].astype(str).str.cat(df['article'].astype(str), sep=' ')
    if text_length or digit_density:
        lengths = texts.str.len()
        if text_length:
            df_dict['text_length'] = lengths
        if digit_density:
            digit_counts = texts.str.count(r'\d')
            digit_densities = digit_counts / (lengths + 1)
            df_dict['digit_density'] = digit_densities
    if title_length:
        title_lengths = df['title'].astype(str).str.len()
        df_dict['title_length'] = title_lengths
    if article_length:
        article_lengths = df['article'].astype(str).str.len()
        df_dict['article_length'] = article_lengths
    cleaner = TextCleaner()
    texts = texts.apply(cleaner.clean_text)
    df_dict['full_text'] = texts
    return pd.DataFrame(df_dict)


def timestamp_encoding(df: pd.Series, date_present=True, year=True, month=True, day=True, hour=True, weekday=True):
    """
    The function extracts the components of the timestamp associated to each article according to
    the elements requested via the boolean parameters.

    :param df:
    :param date_present: A boolean column that signals if the value is originally present or not.
    :param year: Whether to add the 'year_' column or not.
    :param month: Whether to add the 'month-' column or not.
    :param day: Whether to add the 'day_' column or not.
    :param hour: Whether to add the 'hour_' column or not.
    :param weekday: Whether to add the 'weekday_' column or not.
    :return: A DataFrame made by the columns specified by the parameters.
    """
    if not (date_present or year or month or day or hour or weekday):
        raise ValueError('At least one parameter must be True.')
    if isinstance(df, pd.DataFrame):
        s = df.iloc[:, 0]
    else:
        s = df
    timestamps = pd.to_datetime(s, errors='coerce')
    df_enc = pd.DataFrame(index=timestamps.index)
    if date_present:
        df_enc['date_present_'] = timestamps.notna().astype(int)
    if year:
        df_enc['year_'] = timestamps.dt.year
    if month:
        df_enc['month_'] = timestamps.dt.month
    if day:
        df_enc['day_'] = timestamps.dt.day
    if hour:
        df_enc['hour_'] = timestamps.dt.hour
    if weekday:
        df_enc['weekday_'] = timestamps.dt.day_of_week
    return df_enc