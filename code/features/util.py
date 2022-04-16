SPACE_STR = ' '
MIN_DF = 50
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
STOP_WORDS = {
    'aafc', 'ffff', 'ffffe', 'affd', 'fffe',
}
STOP_WORDS_EXTRA = {
    'uid', 'xff',
}
STOP_WORDS = STOP_WORDS.union(STOP_WORDS_EXTRA)
